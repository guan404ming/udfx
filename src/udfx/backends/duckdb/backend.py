from __future__ import annotations

import hashlib
import itertools
import string
import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
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
from udfx.backends.duckdb.rules import RULES, bind, column, constant, function, when
from udfx.backends.faults import NativeFaults, faults_for
from udfx.backends.registration import RuleSet
from udfx.ir import FALSE, Col, Lit
from udfx.program import Program
from udfx.semantics.facts import has_set_operation
from udfx.types import Type

EXACT = f"{RESERVED}exact"
NATIVE = f"{RESERVED}native"
ROW = f"{RESERVED}row"
SLOW_LAMBDAS = 8
FALLBACK = "udfx fallback"
ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)
CONNECTIONS = weakref.WeakKeyDictionary[duckdb.DuckDBPyConnection, int]()
NUMBERS = itertools.count()
ELEMENTS = itertools.count()
LOCK = threading.Lock()
DEFAULT_FAULTS = NativeFaults()


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
    binds: ClassVar[bool] = True

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

    def build(self, program: Program) -> Expression:
        expr = super().build(program)
        if has_set_operation(program.value) or str(expr).count(" -> ") >= SLOW_LAMBDAS:
            return self.fallback(
                self.literal(Lit(None, program.result)), self.literal(FALSE), program
            )
        return expr

    def bind(self, value: Expression, body: Callable[[Expression], Expression]) -> Expression:
        return bind(value, body, f"{RESERVED}element {next(ELEMENTS)}")

    def list_value(self, items: Sequence[Expression], type: Type) -> Expression:
        return function("list_value", *items).cast(dtype(type))

    def list_transform(
        self, source: Expression, transform: Callable[[Expression], Expression]
    ) -> Expression:
        name = f"{RESERVED}element {next(ELEMENTS)}"
        return function(
            "list_transform", source, duckdb.LambdaExpression(name, transform(column(name)))
        )

    def list_filter(
        self, source: Expression, predicate: Callable[[Expression], Expression]
    ) -> Expression:
        name = f"{RESERVED}element {next(ELEMENTS)}"
        return function(
            "list_filter", source, duckdb.LambdaExpression(name, predicate(column(name)))
        )

    @property
    def native_faults(self) -> NativeFaults:
        return DEFAULT_FAULTS if self._connection is None else faults_for(self._connection)

    def native_raise(self, program: Program) -> Expression:
        packed = (
            function(
                "struct_pack",
                *(column(name).alias(str(index)) for index, name in enumerate(program.schema)),
            )
            if program.schema
            else constant({})
        )
        encoded = function("hex", function("encode", function("to_json", packed)))
        message = function("concat", constant(self.native_faults.marker(program)), encoded)
        return function("error", message).cast(dtype(program.result))

    def fallback(self, native: Expression, exact: Expression, program: Program) -> Expression:
        connection = self.connection
        name = fallback_name(program, connection)
        fields = {
            EXACT: sqltypes.BOOLEAN,
            NATIVE: dtype(program.result),
            **{field: dtype(type) for field, type in program.schema.items()},
        }

        def patch(packed: pa.ChunkedArray) -> pa.Array:
            batch = cast(pa.StructArray, packed.combine_chunks())
            columns = [batch.field(field) for field in program.schema]
            return arrow.fallback(
                DuckDBBackend(), program, batch.field(EXACT), batch.field(NATIVE), columns
            )

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
            constant(None).cast(dtype(program.result)).alias(NATIVE),
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
