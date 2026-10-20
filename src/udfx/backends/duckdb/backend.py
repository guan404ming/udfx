from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import count
from typing import ClassVar, cast

import duckdb
import pyarrow as pa
from duckdb import Expression, sqltypes
from duckdb.func import ARROW, SPECIAL

from udfx.backends import arrow
from udfx.backends.base import Backend, Row, RuleSet
from udfx.backends.duckdb.dtypes import dtype
from udfx.backends.duckdb.rules import NAN, RULES, column, constant, function, when
from udfx.ir import Col, Lit
from udfx.program import Program
from udfx.types import Type

EXACT = "udfx exact"
OUT = "udfx out"
ROW = "udfx row"
FALLBACK = "udfx fallback"
COUNTER = count()


class DuckDBBackend(Backend[Expression]):
    name: ClassVar[str] = "duckdb"
    rules: ClassVar[RuleSet[Expression]] = RULES

    def __init__(self, connection: duckdb.DuckDBPyConnection | None = None) -> None:
        self.connection = connection or duckdb.connect()

    def column(self, node: Col) -> Expression:
        return column(node.name)

    def literal(self, node: Lit) -> Expression:
        if isinstance(node.value, float) and math.isnan(node.value):
            return NAN
        return constant(node.value).cast(dtype(node.type))

    def select(self, condition: Expression, then: Expression, orelse: Expression) -> Expression:
        return when(condition, then, orelse)

    def both(self, left: Expression, right: Expression) -> Expression:
        return left & right

    def negate(self, value: Expression) -> Expression:
        return ~value

    def fallback(self, native: Expression, exact: Expression, program: Program) -> Expression:
        name = f"{FALLBACK} {next(COUNTER)}"
        fields = {
            EXACT: sqltypes.BOOLEAN,
            **{field: dtype(type) for field, type in program.schema.items()},
        }

        result = arrow.dtype(program.result)

        def patch(packed: pa.ChunkedArray) -> pa.Array:
            batch = cast(pa.StructArray, packed.combine_chunks())
            flags = batch.field(EXACT).fill_null(False).to_pylist()
            pending = [index for index, flag in enumerate(flags) if not flag]
            values: list[object] = [None] * len(batch)
            if pending:
                columns = [batch.field(field).take(pending).to_pylist() for field in program.schema]
                patched = self.patch(program, zip(*columns, strict=True))
                for index, value in zip(pending, patched, strict=True):
                    values[index] = value
            return pa.array(values, result)

        self.connection.create_function(
            name,
            patch,
            [duckdb.struct_type(fields)],
            dtype(program.result),
            type=ARROW,
            null_handling=SPECIAL,
        )
        packed = function(
            "struct_pack",
            exact.alias(EXACT),
            *(column(field).alias(field) for field in program.schema),
        )
        return when(exact, native, function(name, packed))

    def evaluate(
        self, exprs: Sequence[Expression], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]:
        columns = list(zip(*rows, strict=True)) if rows else [() for _ in schema]
        table = pa.table(
            {
                ROW: pa.array(range(len(rows)), pa.int64()),
                **{
                    name: pa.array(list(values))
                    for name, values in zip(schema, columns, strict=True)
                },
            }
        )
        typed = self.connection.from_arrow(table).select(
            column(ROW),
            *(column(name).cast(dtype(type)).alias(name) for name, type in schema.items()),
        )
        names = [f"{OUT} {index}" for index in range(len(exprs))]
        selected = typed.select(
            *(expr.alias(name) for expr, name in zip(exprs, names, strict=True))
        )
        try:
            outputs = selected.fetchall()
        except Exception as wrapped:
            raise self.unwrap(wrapped) from None
        return [list(values) for values in zip(*outputs, strict=True)] or [[] for _ in exprs]
