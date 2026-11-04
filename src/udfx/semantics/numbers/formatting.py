import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from udfx.ir import Call, Lit
from udfx.semantics.bindings import source
from udfx.semantics.numbers.operations import (
    EXACT_POWERS_OF_TEN,
    FLOAT_INT_MAX,
    HALF_INTEGER_LIMIT,
    RoundPrimitives,
    scaled_round,
)
from udfx.semantics.numbers.ranges import integer_bounds
from udfx.types import INT64_MAX, Kind


def fixed_format(value: float, digits: int) -> str:
    return format(value, f".{digits}f")


def fixed_precision(value: object) -> int | None:
    return value if isinstance(value, int) and 0 <= value <= EXACT_POWERS_OF_TEN else None


def fixed_format_exact(node: Call) -> bool:
    value, precision = map(source, node.arguments)
    digits = fixed_precision(precision.value) if isinstance(precision, Lit) else None
    if digits is None or not isinstance(value, Call) or value.op.name != "__mul__":
        return False
    left, right = map(source, value.arguments)
    if left.type.kind is not Kind.INT:
        left, right = right, left
    if (
        left.type.kind is not Kind.INT
        or not isinstance(right, Lit)
        or not isinstance(right.value, float)
        or not math.isfinite(right.value)
    ):
        return False
    bounds = integer_bounds(left, INT64_MAX)
    if bounds is None or (maximum := max(map(abs, bounds))) > FLOAT_INT_MAX:
        return False
    scale = 10**digits
    factor = Fraction(right.value)
    unit = Fraction(sys.float_info.epsilon)
    minimum = Fraction(1, 2**1074)
    first = maximum * abs(factor)
    first_error = first * unit + minimum
    second_error = (first + first_error) * scale * unit + minimum
    error = first_error * scale + second_error
    distance = maximum * abs(factor * scale - round(factor * scale)) + error
    return first * scale + error < HALF_INTEGER_LIMIT and (digits == 0 or distance < Fraction(1, 2))


@dataclass(frozen=True)
class FormatPrimitives:
    rounding: RoundPrimitives
    half_even: Callable[[Any], Any]
    integer_text: Callable[[Any, int], Any]
    insert_decimal: Callable[[Any, str], Any]
    concat: Callable[..., Any]
    negative: Callable[[Any], Any]


def format_fixed(
    primitives: FormatPrimitives, value: Any, digits: Any, *, exact: bool = False
) -> tuple[Any, Any]:
    rounding = primitives.rounding
    literal = rounding.literal
    digits = fixed_precision(digits)
    if digits is None:
        return literal(""), literal(False)
    scaled = scaled_round(rounding, value, digits)
    safe = rounding.when(scaled.exact, rounding.abs(scaled.scaled), literal(0.0))
    rounded = primitives.half_even(safe) if digits == 0 else rounding.round(safe)
    text = primitives.integer_text(rounded, digits + 1)
    if digits:
        text = primitives.insert_decimal(text, rf"([0-9]{{{digits}}})$")
    sign = rounding.when(primitives.negative(value), literal("-"), literal(""))
    return primitives.concat(sign, text), None if exact else scaled.exact
