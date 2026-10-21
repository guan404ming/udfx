from udfx.backends.duckdb import conversions, core, math, numbers, regex, strings  # noqa: F401  # pyright: ignore[reportUnusedImport]
from udfx.backends.duckdb.backend import DuckDBBackend

BACKEND = DuckDBBackend

__all__ = ["BACKEND", "DuckDBBackend"]
