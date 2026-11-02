from pyspark.sql import Column, functions

from udfx.backends.registration import Lowered
from udfx.backends.spark.math import ROUND
from udfx.backends.spark.rules import RULES
from udfx.semantics.formatting import FormatPrimitives, fixed_format_exact, format_fixed
from udfx.types import Kind

PRIMITIVES = FormatPrimitives(
    ROUND,
    functions.rint,
    lambda value, width: functions.format_string(f"%0{width}d", value.cast("long")),
    lambda value, pattern: functions.regexp_replace(value, pattern, ".$1"),
    functions.concat,
    lambda value: value.cast("string").startswith("-"),
)


@RULES("format_fixed", Kind.FLOAT, Kind.INT, literals=(1,), facts={"exact": fixed_format_exact})
def fixed(value: Column, digits: Column | int, *, exact: bool) -> Lowered[Column]:
    return Lowered(*format_fixed(PRIMITIVES, value, digits, exact=exact))
