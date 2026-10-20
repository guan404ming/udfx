from udfx.backends.datafusion import core, math, numbers, strings
from udfx.backends.datafusion.backend import DataFusionBackend

BACKEND = DataFusionBackend

__all__ = ["BACKEND", "DataFusionBackend", "core", "math", "numbers", "strings"]
