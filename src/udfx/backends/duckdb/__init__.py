# pyright: reportUnusedImport=false

from udfx.backends.duckdb import (  # noqa: F401
    conversions,
    core,
    datetimes,
    json,
    lists,
    math,
    numbers,
    regex,
    slicing,
    splitting,
    strings,
)
from udfx.backends.duckdb.backend import DuckDBBackend

BACKEND = DuckDBBackend

__all__ = ["BACKEND", "DuckDBBackend"]
