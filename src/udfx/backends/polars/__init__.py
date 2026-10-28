# pyright: reportUnusedImport=false

from udfx.backends.polars import (  # noqa: F401
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
from udfx.backends.polars.backend import PolarsBackend

BACKEND = PolarsBackend

__all__ = ["BACKEND", "PolarsBackend"]
