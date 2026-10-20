from udfx.backends.duckdb import core, math, numbers, strings  # noqa: F401  # pyright: ignore[reportUnusedImport]
from udfx.backends.duckdb.backend import DuckDBBackend

BACKEND = DuckDBBackend

__all__ = ["BACKEND", "DuckDBBackend"]
