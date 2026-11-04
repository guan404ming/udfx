import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.engines.datafusion.math import ROUND, half_even
from udfx.backends.engines.datafusion.rules import RULES
from udfx.backends.registration.shared import Lowered
from udfx.semantics.numbers.formatting import FormatPrimitives, fixed_format_exact, format_fixed
from udfx.types import Kind


def integer_text(value: Expr, width: int) -> Expr:
    text = value.cast(pa.int64()).cast(pa.string())
    return functions.lpad(
        text, functions.greatest(functions.character_length(text), lit(width)), "0"
    )


PRIMITIVES = FormatPrimitives(
    ROUND,
    half_even,
    integer_text,
    lambda value, pattern: functions.regexp_replace(value, pattern, ".${1}"),
    functions.concat,
    lambda value: functions.starts_with(value.cast(pa.string()), lit("-")),
)


@RULES("format_fixed", Kind.FLOAT, Kind.INT, literals=(1,), facts={"exact": fixed_format_exact})
def fixed(value: Expr, digits: Expr | int, *, exact: bool) -> Lowered[Expr]:
    return Lowered(*format_fixed(PRIMITIVES, value, digits, exact=exact))
