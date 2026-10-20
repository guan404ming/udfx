from udfx.backends.duckdb import core, math, numbers, strings
from udfx.backends.duckdb.backend import DuckDBBackend

BACKEND = DuckDBBackend

__all__ = ["BACKEND", "DuckDBBackend", "core", "math", "numbers", "strings"]
