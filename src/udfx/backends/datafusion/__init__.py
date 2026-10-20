from udfx.backends.datafusion import core, numbers, strings
from udfx.backends.datafusion.backend import DataFusionBackend

BACKEND = DataFusionBackend

__all__ = ["BACKEND", "DataFusionBackend", "core", "numbers", "strings"]
