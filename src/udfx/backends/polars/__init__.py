from udfx.backends.polars import core, math, numbers, strings
from udfx.backends.polars.backend import PolarsBackend

BACKEND = PolarsBackend

__all__ = ["BACKEND", "PolarsBackend", "core", "math", "numbers", "strings"]
