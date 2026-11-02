from __future__ import annotations

import itertools
import threading
import weakref
from contextlib import suppress
from contextvars import ContextVar

import duckdb
import pyarrow as pa
from duckdb import Expression
from duckdb.func import ARROW

from udfx.backends.base import RESERVED
from udfx.semantics.scalars import numpy_values

CONNECTION: ContextVar[duckdb.DuckDBPyConnection | None] = ContextVar("connection", default=None)
CONNECTIONS = weakref.WeakKeyDictionary[duckdb.DuckDBPyConnection, int]()
NUMBERS = itertools.count()
LOCK = threading.Lock()


def connection_number(connection: duckdb.DuckDBPyConnection) -> int:
    with LOCK:
        return CONNECTIONS.setdefault(connection, next(NUMBERS))


def numpy_function(name: str, value: Expression) -> Expression:
    connection = CONNECTION.get() or duckdb.default_connection()
    function = f"{RESERVED}numpy {name} {connection_number(connection)}"

    def run(values: pa.ChunkedArray) -> pa.Array:
        return pa.array(numpy_values(name, values.to_numpy()), pa.float64())

    with suppress(duckdb.NotImplementedException):
        connection.create_function(function, run, ["DOUBLE"], "DOUBLE", type=ARROW)
    return duckdb.FunctionExpression(function, value)
