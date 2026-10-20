from __future__ import annotations

import math
import string
from collections.abc import Mapping, Sequence
from typing import ClassVar, cast

import duckdb
import pyarrow as pa
from duckdb import Expression, sqltypes
from duckdb.func import ARROW, SPECIAL

from udfx.backends import arrow
from udfx.backends.base import RESERVED, Backend, Row, RuleSet, output_names, transpose_rows
from udfx.backends.duckdb.dtypes import dtype
from udfx.backends.duckdb.rules import NAN, RULES, column, constant, function, when
from udfx.ir import Col, Lit
from udfx.program import Program
from udfx.types import Type

EXACT = f"{RESERVED}exact"
ROW = f"{RESERVED}row"
FALLBACK = "udfx fallback"
ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


class DuckDBBackend(Backend[Expression]):
    name: ClassVar[str] = "duckdb"
    rules: ClassVar[RuleSet[Expression]] = RULES

    def __init__(self, connection: duckdb.DuckDBPyConnection | None = None) -> None:
        self._connection = connection
        self.registered: set[tuple[duckdb.DuckDBPyConnection, str]] = set()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.default_connection() if self._connection is None else self._connection

    def fold_name(self, name: str) -> str:
        return name.translate(ASCII_LOWER)

    def column(self, node: Col) -> Expression:
        return column(node.name)

    def literal(self, node: Lit) -> Expression:
        if isinstance(node.value, float) and math.isnan(node.value):
            return NAN
        return constant(node.value).cast(dtype(node.type))

    def select(self, condition: Expression, then: Expression, orelse: Expression) -> Expression:
        return when(condition, then, orelse)

    def fallback(self, native: Expression, exact: Expression, program: Program) -> Expression:
        name = f"{FALLBACK} {id(self)} {id(program)}"
        fields = {
            EXACT: sqltypes.BOOLEAN,
            **{field: dtype(type) for field, type in program.schema.items()},
        }

        result = arrow.dtype(program.result)

        def patch(packed: pa.ChunkedArray) -> pa.Array:
            batch = cast(pa.StructArray, packed.combine_chunks())
            columns = [batch.field(field) for field in program.schema]
            return arrow.fallback(self, program, batch.field(EXACT), columns, result)

        connection = self.connection
        if (connection, name) not in self.registered:
            connection.create_function(
                name,
                patch,
                [duckdb.struct_type(fields)],
                dtype(program.result),
                type=ARROW,
                null_handling=SPECIAL,
            )
            self.registered.add((connection, name))
        packed = function(
            "struct_pack",
            exact.alias(EXACT),
            *(column(field).alias(field) for field in program.schema),
        )
        return when(exact, native, function(name, packed))

    def evaluate(
        self, exprs: Sequence[Expression], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]:
        self.check_names(schema)
        columns = transpose_rows(rows, len(schema))
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
        names = output_names(len(exprs))
        aliased = [expr.alias(name) for expr, name in zip(exprs, names, strict=True)]
        outputs = self.execute(lambda: typed.select(*aliased).fetchall())
        return [list(values) for values in zip(*outputs, strict=True)] or [[] for _ in exprs]
