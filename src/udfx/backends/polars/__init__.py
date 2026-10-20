from udfx.backends.polars import core, numbers, strings
from udfx.backends.polars.backend import PolarsBackend

BACKEND = PolarsBackend

__all__ = ["BACKEND", "PolarsBackend", "core", "numbers", "strings"]
