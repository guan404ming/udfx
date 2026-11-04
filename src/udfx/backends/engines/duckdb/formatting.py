from duckdb import Expression

from udfx.backends.engines.duckdb.math import ROUND
from udfx.backends.engines.duckdb.rules import RULES, constant, function
from udfx.backends.registration.shared import Lowered
from udfx.semantics.numbers.formatting import fixed_format_exact, fixed_precision
from udfx.semantics.numbers.operations import scaled_round
from udfx.types import Kind


@RULES("format_fixed", Kind.FLOAT, Kind.INT, literals=(1,), facts={"exact": fixed_format_exact})
def fixed(value: Expression, digits: Expression | int, *, exact: bool) -> Lowered[Expression]:
    precision = fixed_precision(digits)
    if precision is None:
        return Lowered(constant(""), constant(False))
    return Lowered(
        function("printf", constant(f"%.{precision}f"), value),
        None if exact else scaled_round(ROUND, value, precision).exact,
    )
