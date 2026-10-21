from udfx.backends.polars import conversions, core, datetimes, lists, math, numbers, regex, strings  # noqa: F401  # pyright: ignore[reportUnusedImport]
from udfx.backends.polars.backend import PolarsBackend

BACKEND = PolarsBackend

__all__ = ["BACKEND", "PolarsBackend"]
