from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import ClassVar

import polars as pl

from udfx.backends.base import RAISED, RESERVED, Backend, Row, output_names, transpose_rows
from udfx.backends.polars.dtypes import dtype
from udfx.backends.polars.rules import RULES, broadcast, evaluate_list
from udfx.backends.registration import Lowered, RuleSet
from udfx.errors import Unsupported
from udfx.ir import Col, Comprehension, Element, Lit, Node
from udfx.program import Program
from udfx.types import Kind, Type

ROW = f"{RESERVED}row"


class PolarsBackend(Backend[pl.Expr]):
    name: ClassVar[str] = "polars"
    rules: ClassVar[RuleSet[pl.Expr]] = RULES

    def build(self, program: Program) -> pl.Expr:
        expr = super().build(program)
        return expr if expr.meta.root_names() else broadcast(expr)

    def lower(
        self, node: Node, elements: Mapping[Element, pl.Expr] | None = None
    ) -> Lowered[pl.Expr]:
        if isinstance(node, Comprehension) and elements:
            raise Unsupported("Polars cannot nest list element scopes")
        return super().lower(node, elements)

    def run_fallback_rows(self, program: Program, rows: Iterable[Sequence[object]]) -> list[object]:
        try:
            return program.run(rows)
        except Exception as error:
            RAISED.error = error
            raise

    def column(self, node: Col) -> pl.Expr:
        return pl.col(node.name)

    def literal(self, node: Lit) -> pl.Expr:
        return pl.lit(node.value, dtype=dtype(node.type))

    def select(self, condition: pl.Expr, then: pl.Expr, orelse: pl.Expr) -> pl.Expr:
        return pl.when(condition).then(then).otherwise(orelse)

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

    def fallback(self, native: pl.Expr, exact: pl.Expr, program: Program) -> pl.Expr:
        result = dtype(program.result)
        flag_name, value_name = output_names(2)
        expressions = [exact.alias(flag_name), native.alias(value_name)]

        def patch(series: Sequence[pl.Series]) -> pl.Series:
            length = len(series[0])
            outputs = (
                pl.DataFrame(series)
                .lazy()
                .select(expressions)
                .collect(engine="in-memory")
                .get_columns()
            )
            flags, values = (
                item if len(item) == length else item.new_from_index(0, length) for item in outputs
            )
            pending = (~flags.fill_null(False)).arg_true().to_list()

            def gather(rows: list[int]) -> list[list[object]]:
                return (
                    [column.gather(rows).to_list() for column in series] if program.schema else []
                )

            if not pending:
                return values
            patched = self.fallback_values(program, length, pending, gather)
            return values.zip_with(flags.fill_null(False), pl.Series(patched, dtype=result))

        inputs = [pl.col(name) for name in program.schema] or [pl.int_range(pl.len()).alias(ROW)]
        return pl.map_batches(inputs, patch, return_dtype=result, is_elementwise=True)

    def evaluate(
        self, exprs: Sequence[pl.Expr], schema: Mapping[str, Type], rows: Sequence[Row]
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
