from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar

import polars as pl

from udfx.backends.base import RAISED, RESERVED, Backend, Row, output_names, transpose_rows
from udfx.backends.polars.dtypes import dtype
from udfx.backends.polars.rules import RULES, broadcast, evaluate_list
from udfx.backends.registration import Lowered, RuleSet
from udfx.backends.rules import Scope
from udfx.errors import Unsupported
from udfx.graphs import free
from udfx.ir import Col, Comprehension, Lit, Node
from udfx.program import Program
from udfx.types import Kind, Type

ROW = f"{RESERVED}row"


class PolarsBackend(Backend[pl.Expr]):
    name: ClassVar[str] = "polars"
    rules: ClassVar[RuleSet[pl.Expr]] = RULES
    graphs: ClassVar[bool] = True

    def build(self, program: Program) -> pl.Expr:
        expr = super().build(program)
        return expr if expr.meta.root_names() else broadcast(expr)

    def lower(self, node: Node, elements: Scope[pl.Expr] | None = None) -> Lowered[pl.Expr]:
        if isinstance(node, Comprehension) and elements:
            captured = free(node.value) | free(node.condition) | free(node.fault)
            if captured - {node.element}:
                raise Unsupported("Polars cannot capture outer list elements")
        return super().lower(node, elements)

    def run_fallback[T](self, run: Callable[[], T]) -> T:
        try:
            return run()
        except Exception as error:
            RAISED.error = error
            raise

    def column(self, node: Col) -> pl.Expr:
        return pl.col(node.name)

    def bind(self, value: pl.Expr, body: Callable[[pl.Expr], pl.Expr]) -> pl.Expr:
        return pl.concat_arr(value).arr.to_list().list.eval(body(pl.element())).list.first()

    def bind_sequence(
        self, value: pl.Expr, steps: Sequence[Callable[[pl.Expr], pl.Expr]]
    ) -> pl.Expr:
        def nested(index: int, expression: pl.Expr) -> pl.Expr:
            if index == len(steps):
                return expression
            return self.bind(expression, lambda item: nested(index + 1, steps[index](item)))

        return nested(0, value)

    def record(self, values: Sequence[pl.Expr]) -> pl.Expr:
        return pl.struct([value.alias(str(index)) for index, value in enumerate(values)])

    def field(self, value: pl.Expr, index: int) -> pl.Expr:
        return value.struct.field(str(index))

    def literal(self, node: Lit) -> pl.Expr:
        return pl.lit(node.value, dtype=dtype(node.type))

    def select(self, condition: pl.Expr, then: pl.Expr, orelse: pl.Expr) -> pl.Expr:
        return pl.when(condition).then(then).otherwise(orelse)

    def select_many(self, cases: Sequence[tuple[pl.Expr, pl.Expr]], orelse: pl.Expr) -> pl.Expr:
        condition, value = cases[0]
        selected = pl.when(condition).then(value)
        for condition, value in cases[1:]:
            selected = selected.when(condition).then(value)
        return selected.otherwise(orelse)

    def list_value(self, items: Sequence[pl.Expr], type: Type) -> pl.Expr:
        if items and type.item is not None and type.item.kind is Kind.LIST:
            return pl.concat_arr(items).arr.to_list().cast(dtype(type))
        return pl.concat_list(items).cast(dtype(type)) if items else self.literal(Lit((), type))

    def list_transform(self, source: pl.Expr, transform: Callable[[pl.Expr], pl.Expr]) -> pl.Expr:
        element = pl.element()
        body = transform(element)
        return evaluate_list(source, broadcast(body))

    def list_filter(self, source: pl.Expr, predicate: Callable[[pl.Expr], pl.Expr]) -> pl.Expr:
        return evaluate_list(source, pl.element().filter(predicate(pl.element())))

    def list_indexed(
        self,
        source: pl.Expr,
        value: Callable[[pl.Expr, pl.Expr], pl.Expr],
        condition: Callable[[pl.Expr, pl.Expr], pl.Expr] | None,
    ) -> pl.Expr:
        element, index = pl.element(), pl.int_range(pl.len())
        body = broadcast(value(element, index))
        if condition is not None:
            body = body.filter(broadcast(condition(element, index)))
        return evaluate_list(source, body)

    def compose(self, value: Lowered[pl.Expr], program: Program) -> pl.Expr:
        if value.prepare is None or value.exact is None or value.exact is False:
            return super().compose(value, program)
        packed = value.consume(lambda part: self.record([self.exact_expression(part), part.expr]))
        return self.fallback_record(packed, program)

    def fallback_record(self, packed: pl.Expr, program: Program) -> pl.Expr:
        result = dtype(program.result)
        name = output_names(1)[0]

        def patch(series: Sequence[pl.Series]) -> pl.Series:
            length = len(series[0])
            outputs = (
                pl.DataFrame(series)
                .rechunk()
                .lazy()
                .select(packed.alias(name))
                .collect(engine="in-memory")
                .get_column(name)
            )
            flags, values = (
                item if len(item) == length else item.new_from_index(0, length)
                for item in (outputs.struct.field("0"), outputs.struct.field("1"))
            )
            missing = (~flags.fill_null(False)).arg_true()
            if program.series is not None and len(missing):
                everything = len(missing) == length
                columns = series if everything else [column.gather(missing) for column in series]
                found = pl.Series(
                    self.run_fallback(
                        lambda: program.batch([column.to_pandas() for column in columns])
                    ),
                    dtype=result,
                )
                return found if everything else values.scatter(missing, found)
            pending = missing.to_list()

            def gather(rows: Sequence[int]) -> list[list[object]]:
                indices = rows if isinstance(rows, list) else list(rows)
                return (
                    [column.gather(indices).to_list() for column in series]
                    if program.schema
                    else []
                )

            if not pending:
                return values
            patched = self.fallback_values(program, length, pending, gather)
            return values.zip_with(flags.fill_null(False), pl.Series(patched, dtype=result))

        inputs = [pl.col(name) for name in program.schema] or [pl.int_range(pl.len()).alias(ROW)]
        return pl.map_batches(
            inputs, patch, return_dtype=result, is_elementwise=program.deterministic
        )

    def fallback(self, native: pl.Expr, exact: pl.Expr, program: Program) -> pl.Expr:
        return self.fallback_record(self.record([exact, native]), program)

    def evaluate(
        self,
        exprs: Sequence[pl.Expr],
        schema: Mapping[str, Type],
        rows: Sequence[Row],
        reduced: bool = False,
    ) -> list[list[object]]:
        self.check_names(schema)
        columns = transpose_rows(rows, len(schema))
        frame = pl.DataFrame(
            [
                pl.Series(ROW, range(len(rows))),
                *(
                    pl.Series(name, list(values), dtype=dtype(type), strict=True)
                    for (name, type), values in zip(schema.items(), columns, strict=True)
                ),
            ]
        )
        names = output_names(len(exprs))
        aliased = [expr.alias(name) for expr, name in zip(exprs, names, strict=True)]
        outputs = self.execute(lambda: frame.select(*aliased))
        return [outputs.get_column(name).to_list() for name in names]
