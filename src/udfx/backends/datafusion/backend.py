from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from itertools import count
from typing import ClassVar, cast

import pyarrow as pa
from datafusion import Expr, SessionContext, functions, lit, udf

from udfx.backends import arrow
from udfx.backends.arrow import dtype
from udfx.backends.base import RESERVED, Backend, Row, output_names, transpose_rows
from udfx.backends.datafusion import graphs
from udfx.backends.datafusion.rules import (
    ARRAY_SCOPING,
    RULES,
    array_filter,
    array_lambda,
    array_transform,
    column,
    typed_literal,
)
from udfx.backends.lowering_context import graph_enabled, lowering_context
from udfx.backends.program_lowering import lower_program
from udfx.backends.registration import Lowered, RuleSet
from udfx.backends.results import FallbackLowering, ResultLowering, fallback_result
from udfx.backends.rules import Scope
from udfx.ir import Col, Comprehension, Element, Lit, Node
from udfx.program import Program
from udfx.types import Kind, Type

ROW = f"{RESERVED}row"
FALLBACKS = count()


class DataFusionBackend(Backend[Expr]):
    name: ClassVar[str] = "datafusion"
    rules: ClassVar[RuleSet[Expr]] = RULES
    graphs: ClassVar[bool] = True

    def __init__(self, context: SessionContext | None = None) -> None:
        self._context = context
        self._graphs: bool | None = None

    def supports_graphs(self) -> bool:
        if self._graphs is None:
            self._graphs = graphs.supported(self.context)
        return self._graphs

    @property
    def context(self) -> SessionContext:
        if self._context is None:
            self._context = SessionContext()
        return self._context

    def column(self, node: Col) -> Expr:
        return column(node.name)

    def lower_program(
        self, program: Program, results: ResultLowering[Expr] | None = None
    ) -> Lowered[Expr]:
        supported = self.supports_graphs()
        token = ARRAY_SCOPING.set(not supported)
        try:
            with lowering_context(binds=supported):
                return lower_program(self, program, results)
        finally:
            ARRAY_SCOPING.reset(token)

    def compose(self, value: Lowered[Expr], program: Program) -> Expr:
        value = fallback_result(
            value,
            FallbackLowering(self, program),
            lambda results: self.lower_program(program, results),
        )
        return super().compose(value, program)

    def lower(self, node: Node, elements: Scope[Expr] | None = None) -> Lowered[Expr]:
        lowered = super().lower(node, elements)
        return (
            lowered.map(
                lambda value: replace(
                    value, expr=value.expr.cast(dtype(node.type, string_type=pa.string_view()))
                )
            )
            if isinstance(node, Comprehension | Element) and node.type.kind is Kind.LIST
            else lowered
        )

    def literal(self, node: Lit) -> Expr:
        return typed_literal(node.value, node.type)

    def select(self, condition: Expr, then: Expr, orelse: Expr) -> Expr:
        return functions.when(condition, then).otherwise(orelse)

    def select_many(self, cases: Sequence[tuple[Expr, Expr]], orelse: Expr) -> Expr:
        selected = functions.when(*cases[0])
        for condition, value in cases[1:]:
            selected = selected.when(condition, value)
        return selected.otherwise(orelse)

    def select_value(self, value: Expr, cases: Sequence[tuple[Expr, Expr]], orelse: Expr) -> Expr:
        selected = functions.case(value)
        for key, result in cases:
            selected = selected.when(key, result)
        return selected.otherwise(orelse)

    def list_value(self, items: Sequence[Expr], type: Type) -> Expr:
        return (
            functions.make_array(*items).cast(dtype(type)) if items else self.literal(Lit((), type))
        )

    def list_transform(self, source: Expr, transform: Callable[[Expr], Expr]) -> Expr:
        if graph_enabled():
            return functions.array_transform(source, array_lambda(transform))
        return array_transform(source, transform)

    def list_filter(self, source: Expr, predicate: Callable[[Expr], Expr]) -> Expr:
        if graph_enabled():
            return functions.array_filter(source, array_lambda(predicate))
        return array_filter(source, predicate)

    def record(self, values: Sequence[Expr]) -> Expr:
        return functions.named_struct([(str(index), value) for index, value in enumerate(values)])

    def field(self, value: Expr, index: int) -> Expr:
        return value[str(index)]

    def bind(self, value: Expr, body: Callable[[Expr], Expr]) -> Expr:
        return graphs.bind(value, body)

    def bind_sequence(self, value: Expr, steps: Sequence[Callable[[Expr], Expr]]) -> Expr:
        return graphs.bind_sequence(value, steps)

    def fallback(self, native: Expr, exact: Expr, program: Program) -> Expr:
        result = dtype(program.result)
        fields = pa.struct(
            [(name, dtype(type)) for name, type in program.schema.items()] or [(ROW, pa.bool_())]
        )

        def run(packed: pa.Array) -> pa.Array:
            batch = cast(pa.StructArray, packed)
            columns = [batch.field(name) for name in program.schema]
            values = self.evaluate_pending_rows(
                program, range(len(batch)), arrow.make_gather(columns)
            )
            return pa.array(values, type=result)

        slow = udf(
            run,
            [fields],
            result,
            "volatile",
            f"udfx fallback {next(FALLBACKS)}",
        )
        packed = functions.named_struct(
            [(name, column(name)) for name in program.schema] or [(ROW, lit(False))]
        )
        return functions.when(exact, native).otherwise(slow(packed.cast(fields)))

    def evaluate(
        self, exprs: Sequence[Expr], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]:
        self.check_names(schema)
        columns = transpose_rows(rows, len(schema))
        table = pa.table(
            {
                ROW: pa.array(range(len(rows)), type=pa.int64()),
                **{
                    name: pa.array(list(values), type=dtype(type))
                    for (name, type), values in zip(schema.items(), columns, strict=True)
                },
            }
        )
        names = output_names(len(exprs))
        aliased = [expr.alias(name) for expr, name in zip(exprs, names, strict=True)]
        frame = self.context.from_arrow(table)
        outputs = self.execute(
            lambda: frame.select(column(ROW), *aliased).sort(column(ROW)).to_arrow_table()
        )
        return [cast("list[object]", outputs.column(name).to_pylist()) for name in names]
