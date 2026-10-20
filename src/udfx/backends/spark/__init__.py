from udfx.backends.spark import core
from udfx.backends.spark.backend import SparkBackend

BACKEND = SparkBackend

__all__ = ["BACKEND", "SparkBackend", "core"]
