import polars as pl

from udfx.backends.polars.math import ROUND, half_even
from udfx.backends.polars.rules import RULES
from udfx.backends.registration import Lowered
from udfx.semantics.formatting import FormatPrimitives, fixed_format_exact, format_fixed
from udfx.types import Kind

PRIMITIVES = FormatPrimitives(
    ROUND,
    half_even,
    lambda value, width: value.cast(pl.Int64).cast(pl.String).str.pad_start(width, "0"),
    lambda value, pattern: value.str.replace(pattern, ".${1}"),
    pl.concat_str,
    lambda value: value.cast(pl.String).str.starts_with("-"),
)


@RULES("format_fixed", Kind.FLOAT, Kind.INT, literals=(1,), facts={"exact": fixed_format_exact})
def fixed(value: pl.Expr, digits: pl.Expr | int, *, exact: bool) -> Lowered[pl.Expr]:
    return Lowered(*format_fixed(PRIMITIVES, value, digits, exact=exact))
