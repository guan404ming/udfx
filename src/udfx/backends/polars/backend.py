from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

import polars as pl

from udfx.backends.base import Backend, Row, RuleSet
from udfx.backends.polars.dtypes import dtype
from udfx.backends.polars.rules import RULES
from udfx.ir import Col, Lit
from udfx.program import Program
from udfx.types import Type

OUT = "udfx out"
ROW = "udfx row"


class PolarsBackend(Backend[pl.Expr]):
    name: ClassVar[str] = "polars"
    rules: ClassVar[RuleSet[pl.Expr]] = RULES

    def column(self, node: Col) -> pl.Expr:
        return pl.col(node.name)

    def literal(self, node: Lit) -> pl.Expr:
        return pl.lit(node.value, dtype=dtype(node.type))

    def select(self, condition: pl.Expr, then: pl.Expr, orelse: pl.Expr) -> pl.Expr:
        return pl.when(condition).then(then).otherwise(orelse)

    def both(self, left: pl.Expr, right: pl.Expr) -> pl.Expr:
        return left & right

    def negate(self, value: pl.Expr) -> pl.Expr:
        return ~value

    def fallback(self, native: pl.Expr, exact: pl.Expr, program: Program) -> pl.Expr:
        result = dtype(program.result)

        def patch(series: Sequence[pl.Series]) -> pl.Series:
            flags, values, *columns = series
            pending = (~flags.fill_null(False)).arg_true()
            if pending.is_empty():
                return values
            rows = (column.gather(pending).to_list() for column in columns)
            return values.scatter(
                pending, pl.Series(program.run(zip(*rows, strict=True)), dtype=result)
            )

        inputs = [exact, native, *(pl.col(name) for name in program.schema)]
        return pl.map_batches(inputs, patch, return_dtype=result, is_elementwise=True)

    def evaluate(
        self, exprs: Sequence[pl.Expr], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]:
        columns = list(zip(*rows, strict=True)) if rows else [() for _ in schema]
        frame = pl.DataFrame(
            [
                pl.Series(ROW, range(len(rows))),
                *(
                    pl.Series(name, list(values), dtype=dtype(type), strict=True)
                    for (name, type), values in zip(schema.items(), columns, strict=True)
                ),
            ]
        )
        names = [f"{OUT} {index}" for index in range(len(exprs))]
        outputs = frame.with_columns(
            *(expr.alias(name) for expr, name in zip(exprs, names, strict=True))
        )
        return [outputs.get_column(name).to_list() for name in names]
