from udfx.backends.spark import core, numbers, strings
from udfx.backends.spark.backend import SparkBackend

BACKEND = SparkBackend

__all__ = ["BACKEND", "SparkBackend", "core", "numbers", "strings"]
