from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from udfx.semantics.numbers.operations import REPR_FIXED_MAX, REPR_FIXED_MIN

SCIENTIFIC_FORMAT = "%se%+03d"


@dataclass(frozen=True)
class FloatTextPrimitives:
    literal: Callable[[object], Any]
    when: Callable[[Any, Any, Any], Any]
    concat: Callable[..., Any]
    abs: Callable[[Any], Any]
    is_null: Callable[[Any], Any]
    contains: Callable[[Any, str], Any]
    floor: Callable[[Any], Any]
    format: Callable[..., Any]
    one_digit: Callable[[Any], Any]
    trim_right: Callable[[Any, str], Any]
    before: Callable[[Any, str], Any]
    after: Callable[[Any, str], Any]
    to_int: Callable[[Any], Any]
    to_float: Callable[[Any], Any]


@dataclass(frozen=True)
class FloatSpelling:
    text: Any
    plain: Any
    marker: str
    nan: str
    infinity: str


def float_repr(primitives: FloatTextPrimitives, value: Any, spelling: FloatSpelling) -> Any:
    literal, when, contains, text = (
        primitives.literal,
        primitives.when,
        primitives.contains,
        spelling.text,
    )
    named = contains(text, spelling.nan) | contains(text, spelling.infinity)
    exact = primitives.is_null(value) | ~(named | contains(text, spelling.marker))
    names = when(text == literal(spelling.infinity), literal("inf"), literal("-inf"))
    names = when(text == literal(spelling.nan), literal("nan"), names)
    size = primitives.abs(value)
    fixed = (size >= literal(REPR_FIXED_MIN)) & (size < literal(REPR_FIXED_MAX))
    plain = plain_repr(primitives, value, spelling.plain)
    scientific = scientific_repr(primitives, value, spelling)
    return when(exact, text, when(named, names, when(fixed, plain, scientific)))


def plain_repr(primitives: FloatTextPrimitives, value: Any, text: Any) -> Any:
    literal = primitives.literal
    point = primitives.when(primitives.floor(value) == value, literal("0"), literal(""))
    return primitives.concat(primitives.trim_right(text, "0"), point)


def scientific_repr(primitives: FloatTextPrimitives, value: Any, spelling: FloatSpelling) -> Any:
    mantissa = primitives.trim_right(primitives.before(spelling.text, spelling.marker), ".0")
    exponent = primitives.to_int(primitives.after(spelling.text, spelling.marker))
    general = primitives.format(SCIENTIFIC_FORMAT, mantissa, exponent)
    single = primitives.one_digit(value)
    return primitives.when(primitives.to_float(single) == value, single, general)
