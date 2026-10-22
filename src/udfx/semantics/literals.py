from __future__ import annotations

import sys
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

from udfx.semantics.text import char_class, code_points, whitespace

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
    coalesce: Callable[[Any, Any], Any]
    matches: Callable[[Any, str], Any]
    strip: Callable[[Any, str], Any]
    replace: Callable[[Any, str, str], Any]
    translate: Callable[[Any, str, str], Any]
    char_length: Callable[[Any], Any]
    not_null: Callable[[Any], Any]
    try_int: Callable[[Any], Any]
    try_float: Callable[[Any], Any]


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
def integer_pattern() -> str:
    original, replacement = integer_characters()
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
    middle = digits[:-1] + "_]"
    return f"^{padding}*[+-]?{digits}({middle}*{digits})?{padding}*$"


def literal_matches(primitives: TextPrimitives, text: Any, pattern: str) -> Any:
    return primitives.matches(text, pattern) & ~primitives.matches(text, "__")


def integer_matches(primitives: TextPrimitives, text: Any) -> Any:
    return primitives.when(
        is_ascii(primitives, text),
        literal_matches(primitives, text, INT_LITERAL),
        literal_matches(primitives, text, integer_pattern()),
    )


def normalize_integer(primitives: TextPrimitives, text: Any) -> Any:
    original, replacement = integer_characters()
    return primitives.translate(text, original + "_", replacement)


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


def parse_int(
    primitives: TextPrimitives, text: Any, maximum_length: int | None = None
) -> tuple[Any, Any]:
    ascii_valid = literal_matches(primitives, text, INT_LITERAL)
    ascii_cleaned = primitives.replace(primitives.strip(text, ASCII_SPACE), "_", "")
    cleaned = primitives.strip(normalize_integer(primitives, text), ASCII_SPACE)
    value = primitives.when(
        ascii_valid,
        primitives.try_int(ascii_cleaned),
        primitives.when(
            integer_matches(primitives, text), primitives.try_int(cleaned), primitives.literal(None)
        ),
    )
    if maximum_length is not None and maximum_length <= 18:
        return value, None
    present = value == value  # noqa: PLR0124
    return value, primitives.coalesce(present, ~integer_matches(primitives, text))


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
