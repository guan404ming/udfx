from udfx.backends.datafusion import core, math, numbers, strings  # noqa: F401  # pyright: ignore[reportUnusedImport]
from udfx.backends.datafusion.backend import DataFusionBackend

BACKEND = DataFusionBackend

__all__ = ["BACKEND", "DataFusionBackend"]
