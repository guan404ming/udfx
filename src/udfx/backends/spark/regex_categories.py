from functools import cache
from typing import cast

from pyspark.sql import functions

from udfx.backends.spark.backend import SparkBackend
from udfx.semantics.regex_categories import corrected_category
from udfx.semantics.text import code_points
from udfx.types import STR

PROPERTIES = {
    "CATEGORY_DIGIT": r"\p{Nd}",
    "CATEGORY_SPACE": r"\p{IsWhite_Space}",
    "CATEGORY_WORD": r"\p{L}\p{N}_",
}


def native_category(pattern: str) -> str:
    backend = SparkBackend()
    backend.session()
    expr = functions.regexp_replace(functions.col("point"), "[^" + pattern + "]", "")
    values = backend.evaluate([expr], {"point": STR}, [("".join(code_points()),)])
    return cast(str, values[0][0])


@cache
def category(name: str) -> str:
    positive = name.replace("_NOT_", "_")
    if positive != name:
        return "[^" + category(positive) + "]"
    pattern = PROPERTIES[name]
    return corrected_category(name, pattern, native_category(pattern))
