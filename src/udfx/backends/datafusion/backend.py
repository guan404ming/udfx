from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import count
from typing import ClassVar, cast

import pyarrow as pa
from datafusion import Expr, SessionContext, functions, udf

from udfx.backends import arrow
from udfx.backends.arrow import dtype
from udfx.backends.base import RESERVED, Backend, Row, output_names, transpose_rows
from udfx.backends.datafusion.rules import (
    RULES,
    array_filter,
    array_transform,
    column,
    typed_literal,
)
from udfx.backends.registration import Lowered, RuleSet
from udfx.ir import Col, Comprehension, Lit, Node
from udfx.program import Program
from udfx.types import BOOL, Type

ROW = f"{RESERVED}row"
EXACT = f"{RESERVED}exact"
NATIVE = f"{RESERVED}native"
FALLBACKS = count()


class DataFusionBackend(Backend[Expr]):
    name: ClassVar[str] = "datafusion"
    rules: ClassVar[RuleSet[Expr]] = RULES

    def __init__(self, context: SessionContext | None = None) -> None:
        self._context = context

    @property
    def context(self) -> SessionContext:
        if self._context is None:
            self._context = SessionContext()
        return self._context

    def column(self, node: Col) -> Expr:
        return column(node.name)

    def lower(self, node: Node, elements: Mapping[Node, Expr] | None = None) -> Lowered[Expr]:
        lowered = super().lower(node, elements)
        return (
            Lowered(lowered.expr.cast(dtype(node.type)), lowered.exact, lowered.overflow)
            if isinstance(node, Comprehension)
            else lowered
        )

    def literal(self, node: Lit) -> Expr:
        return typed_literal(node.value, node.type)

    def select(self, condition: Expr, then: Expr, orelse: Expr) -> Expr:
        return functions.when(condition, then).otherwise(orelse)

    def list_value(self, items: Sequence[Expr], type: Type) -> Expr:
        return (
            functions.make_array(*items).cast(dtype(type)) if items else self.literal(Lit((), type))
        )

    def list_transform(self, source: Expr, transform: Callable[[Expr], Expr]) -> Expr:
        return array_transform(source, transform)

    def list_filter(self, source: Expr, predicate: Callable[[Expr], Expr]) -> Expr:
        return array_filter(source, predicate)

    def fallback(self, native: Expr, exact: Expr, program: Program) -> Expr:
        result = dtype(program.result)
        fields = pa.struct(
            [(EXACT, dtype(BOOL)), (NATIVE, result)]
            + [(name, dtype(type)) for name, type in program.schema.items()]
        )

        def run(packed: pa.Array) -> pa.Array:
            batch = cast(pa.StructArray, packed)
            columns = [batch.field(name) for name in program.schema]
            return arrow.fallback(self, program, batch.field(EXACT), batch.field(NATIVE), columns)

        slow = udf(
            run,
            [fields],
            result,
            "volatile",
            f"udfx fallback {next(FALLBACKS)}",
        )
        packed = functions.named_struct(
            [(EXACT, exact), (NATIVE, native), *((name, column(name)) for name in program.schema)]
        )
        return slow(packed.cast(fields))

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
