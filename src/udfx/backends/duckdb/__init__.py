from udfx.backends.duckdb import core
from udfx.backends.duckdb.backend import DuckDBBackend

BACKEND = DuckDBBackend

__all__ = ["BACKEND", "DuckDBBackend", "core"]
