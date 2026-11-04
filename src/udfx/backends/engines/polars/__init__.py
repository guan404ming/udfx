# pyright: reportUnusedImport=false

from udfx.backends.engines.polars import (  # noqa: F401
    conversions,
    core,
    datetimes,
    formatting,
    json,
    lists,
    math,
    numbers,
    regex,
    slicing,
    splitting,
    strings,
)
from udfx.backends.engines.polars.backend import PolarsBackend

BACKEND = PolarsBackend

__all__ = ["BACKEND", "PolarsBackend"]
