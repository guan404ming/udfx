from udfx.backends.duckdb import core, strings
from udfx.backends.duckdb.backend import DuckDBBackend

BACKEND = DuckDBBackend

__all__ = ["BACKEND", "DuckDBBackend", "core", "strings"]
