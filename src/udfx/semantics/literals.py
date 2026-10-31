from __future__ import annotations

import sys
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

from udfx.semantics.text import char_class, code_points, whitespace
from udfx.types import INT64_MAX, INT64_MIN

ASCII_SPACE = "\t\n\x0b\x0c\r "
DIGITS = "[0-9]([0-9_]*[0-9])?"
INT_LITERAL = f"^[{ASCII_SPACE}]*[+-]?{DIGITS}[{ASCII_SPACE}]*$"
FLOAT_LITERAL = (
    f"(?i)^[{ASCII_SPACE}]*[+-]?"
    f"(({DIGITS}(\\.({DIGITS})?)?|\\.{DIGITS})(e[+-]?{DIGITS})?|inf|infinity|nan)"
    f"[{ASCII_SPACE}]*$"
)


@dataclass(frozen=True)
class TextPrimitives:
    literal: Callable[[object], Any]
    when: Callable[[Any, Any, Any], Any]
    coalesce: Callable[..., Any]
    matches: Callable[[Any, str], Any]
    strip: Callable[[Any, str], Any]
    replace: Callable[[Any, str, str], Any]
    translate: Callable[[Any, str, str], Any]
    char_length: Callable[[Any], Any]
    not_null: Callable[[Any], Any]
    try_int: Callable[[Any], Any]
    try_float: Callable[[Any], Any]
    lazy_coalesce: bool = False
    possessive: bool = False
    decimal_pattern: Callable[[], str] | None = None


@cache
def integer_characters() -> tuple[str, str]:
    characters = {
        point: str(unicodedata.decimal(point))
        for point in code_points()
        if point.isdecimal() and not point.isascii()
    }
    for point in whitespace():
        if point in ASCII_SPACE:
            continue
        try:
            int(point + "0")
        except ValueError:
            continue
        characters[point] = " "
    return "".join(characters), "".join(characters.values())


@cache
def integer_pattern(possessive: bool = False, digits: str | None = None) -> str:
    original, replacement = integer_characters()
    if digits is None:
        zeros = [point for point, digit in zip(original, replacement, strict=True) if digit == "0"]
        digits = (
            "[0-9"
            + "".join(f"\\x{{{ord(point):X}}}-\\x{{{ord(point) + 9:X}}}" for point in zeros)
            + "]"
        )
    spaces = ASCII_SPACE + "".join(
        point for point, digit in zip(original, replacement, strict=True) if digit == " "
    )
    padding = char_class(spaces)
    suffix = "+" if possessive else ""
    return rf"\A{padding}*[+-]?{digits}+(?:_{digits}+)*{suffix}{padding}*\z"


def literal_matches(primitives: TextPrimitives, text: Any, pattern: str) -> Any:
    return primitives.matches(text, pattern) & ~primitives.matches(text, "__")


def integer_matches(primitives: TextPrimitives, text: Any) -> Any:
    digits = primitives.decimal_pattern() if primitives.decimal_pattern is not None else None
    return primitives.matches(text, integer_pattern(primitives.possessive, digits))


def is_ascii(primitives: TextPrimitives, text: Any) -> Any:
    return ~primitives.matches(text, r"[^\x00-\x7f]")


def int_parses(
    primitives: TextPrimitives, text: Any, maximum_length: int | None = None
) -> tuple[Any, Any]:
    valid = integer_matches(primitives, text)
    limit = sys.get_int_max_str_digits()
    if limit and (maximum_length is None or maximum_length > limit):
        digits = primitives.replace(primitives.strip(text, whitespace() + "+-"), "_", "")
        valid = valid & (
            (primitives.char_length(text) <= limit) | (primitives.char_length(digits) <= limit)
        )
    return valid, None


def float_parses(primitives: TextPrimitives, text: Any) -> tuple[Any, Any]:
    return literal_matches(primitives, text, FLOAT_LITERAL), is_ascii(primitives, text)


def integer_candidates(primitives: TextPrimitives, text: Any) -> tuple[Any, Any]:
    native = primitives.try_int(text)
    original, replacement = integer_characters()
    cleaned = primitives.translate(text, original + "_", replacement)
    return native, primitives.try_int(cleaned)


def parse_int(
    primitives: TextPrimitives,
    text: Any,
    maximum_length: int | None = None,
    valid: bool = False,
) -> tuple[Any, Any]:
    native, converted = integer_candidates(primitives, text)
    value = (
        primitives.coalesce(native, converted)
        if primitives.lazy_coalesce
        else primitives.when(primitives.not_null(native), native, converted)
    )
    if maximum_length is not None and maximum_length <= 18:
        return value, None
    if valid:
        return value, primitives.not_null(value)
    return value, primitives.not_null(value) | ~integer_matches(primitives, text)


def saturated_integer(text: str) -> int:
    return max(INT64_MIN, min(INT64_MAX, int(text)))


def parse_saturated_int(primitives: TextPrimitives, text: Any) -> Any:
    value, _ = parse_int(primitives, text)
    negative = primitives.matches(primitives.strip(text, whitespace()), "^-")
    boundary = primitives.when(
        negative, primitives.literal(INT64_MIN), primitives.literal(INT64_MAX)
    )
    return primitives.coalesce(value, boundary)


def parse_float(primitives: TextPrimitives, text: Any) -> tuple[Any, Any]:
    cleaned = primitives.replace(primitives.strip(text, ASCII_SPACE), "_", "")
    literal = literal_matches(primitives, text, FLOAT_LITERAL)
    value = primitives.when(literal, primitives.try_float(cleaned), primitives.literal(None))
    return value, is_ascii(primitives, text) & (primitives.not_null(value) | ~literal)


@dataclass(frozen=True)
class SpellingPrimitives:
    literal: Callable[[object], Any]
    when: Callable[[Any, Any, Any], Any]
    is_null: Callable[[Any], Any]
    coalesce: Callable[[Any, Any], Any]


def show_none(primitives: SpellingPrimitives, text: Any) -> Any:
    return primitives.coalesce(text, primitives.literal("None"))


def bool_text(primitives: SpellingPrimitives, value: Any) -> Any:
    literal = primitives.literal
    named = primitives.when(value, literal("True"), literal("False"))
    return primitives.when(primitives.is_null(value), literal("None"), named)
