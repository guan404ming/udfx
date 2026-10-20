from udfx.backends.duckdb import core, numbers, strings
from udfx.backends.duckdb.backend import DuckDBBackend

BACKEND = DuckDBBackend

__all__ = ["BACKEND", "DuckDBBackend", "core", "numbers", "strings"]
