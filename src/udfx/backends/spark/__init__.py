from udfx.backends.spark import core, math, numbers, strings  # noqa: F401  # pyright: ignore[reportUnusedImport]
from udfx.backends.spark.backend import SparkBackend

BACKEND = SparkBackend

__all__ = ["BACKEND", "SparkBackend"]
