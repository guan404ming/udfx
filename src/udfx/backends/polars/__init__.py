from udfx.backends.polars import core, math, numbers, strings  # noqa: F401  # pyright: ignore[reportUnusedImport]
from udfx.backends.polars.backend import PolarsBackend

BACKEND = PolarsBackend

__all__ = ["BACKEND", "PolarsBackend"]
