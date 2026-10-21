from __future__ import annotations

import hashlib
import itertools
import string
import threading
import weakref
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import ClassVar, cast

import duckdb
import pyarrow as pa
from duckdb import Expression, sqltypes
from duckdb.func import ARROW, SPECIAL

from udfx import __version__
from udfx.backends import arrow
from udfx.backends.base import RESERVED, Backend, Row, output_names, transpose_rows
from udfx.backends.duckdb.dtypes import dtype
from udfx.backends.duckdb.rules import RULES, column, constant, function, when
from udfx.backends.registration import RuleSet
from udfx.ir import Col, Lit
from udfx.program import Program
from udfx.types import Type

EXACT = f"{RESERVED}exact"
ROW = f"{RESERVED}row"
FALLBACK = "udfx fallback"
ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)
CONNECTIONS = weakref.WeakKeyDictionary[duckdb.DuckDBPyConnection, int]()
NUMBERS = itertools.count()
LOCK = threading.Lock()


def connection_number(connection: duckdb.DuckDBPyConnection) -> int:
    with LOCK:
        return CONNECTIONS.setdefault(connection, next(NUMBERS))


def fallback_name(program: Program, connection: duckdb.DuckDBPyConnection) -> str:
    code = program.function.__code__
    schema = [(name, str(type)) for name, type in program.schema.items()]
    parts = (
        __version__,
        code.co_code,
        code.co_consts,
        code.co_names,
        schema,
        program.value,
        program.fault,
    )
    digest = hashlib.sha256(repr(parts).encode()).hexdigest()
    return f"{FALLBACK} {digest} {connection_number(connection)}"


class DuckDBBackend(Backend[Expression]):
    name: ClassVar[str] = "duckdb"
    rules: ClassVar[RuleSet[Expression]] = RULES

    def __init__(self, connection: duckdb.DuckDBPyConnection | None = None) -> None:
        self._connection = connection

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.default_connection() if self._connection is None else self._connection

    def fold_name(self, name: str) -> str:
        return name.translate(ASCII_LOWER)

    def column(self, node: Col) -> Expression:
        return column(node.name)

    def literal(self, node: Lit) -> Expression:
        return constant(node.value).cast(dtype(node.type))

    def select(self, condition: Expression, then: Expression, orelse: Expression) -> Expression:
        return when(condition, then, orelse)

    def fallback(self, native: Expression, exact: Expression, program: Program) -> Expression:
        connection = self.connection
        name = fallback_name(program, connection)
        fields = {
            EXACT: sqltypes.BOOLEAN,
            **{field: dtype(type) for field, type in program.schema.items()},
        }

        result = arrow.dtype(program.result)

        def patch(packed: pa.ChunkedArray) -> pa.Array:
            batch = cast(pa.StructArray, packed.combine_chunks())
            columns = [batch.field(field) for field in program.schema]
            return arrow.fallback(DuckDBBackend(), program, batch.field(EXACT), columns, result)

        with suppress(duckdb.NotImplementedException):
            connection.create_function(
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
