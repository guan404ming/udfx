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

EXACT = "__udfx_exact"
OUT = "__udfx_out"


class PolarsBackend(Backend[pl.Expr]):
    name: ClassVar[str] = "polars"
    rules: ClassVar[RuleSet[pl.Expr]] = RULES

    def column(self, col: Col) -> pl.Expr:
        return pl.col(col.name)

    def literal(self, lit: Lit) -> pl.Expr:
        return pl.lit(lit.value, dtype=dtype(lit.type))

    def select(self, cond: pl.Expr, then: pl.Expr, orelse: pl.Expr) -> pl.Expr:
        return pl.when(cond).then(then).otherwise(orelse)

    def both(self, a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return a & b

    def negate(self, a: pl.Expr) -> pl.Expr:
        return ~a

    def fallback(self, native: pl.Expr, exact: pl.Expr, program: Program) -> pl.Expr:
        names = list(program.schema)
        result = dtype(program.result)
        function = program.function

        def run(batch: pl.Series) -> pl.Series:
            mask = batch.struct.field(EXACT).fill_null(False)
            columns = [batch.struct.field(name) for name in names]
            values: list[object] = [None] * len(batch)
            for index in (~mask).arg_true():
                values[index] = function(*(column[index] for column in columns))
            return pl.Series(values, dtype=result)

        packed = pl.struct(exact.alias(EXACT), *(pl.col(name) for name in names))
        slow = packed.map_batches(run, return_dtype=result)
        return pl.when(exact.fill_null(False)).then(native).otherwise(slow)

    def evaluate(
        self, expr: pl.Expr, schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[object]:
        columns = list(zip(*rows, strict=True)) if rows else [() for _ in schema]
        frame = pl.DataFrame(
            [
                pl.Series(name, list(values), dtype=dtype(type), strict=True)
                for (name, type), values in zip(schema.items(), columns, strict=True)
            ]
        )
        return frame.with_columns(expr.alias(OUT)).get_column(OUT).to_list()
