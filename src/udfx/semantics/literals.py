from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ASCII_SPACE = "\t\n\x0b\x0c\r "
DIGITS = "[0-9](_?[0-9])*"
INT_LITERAL = f"^[{ASCII_SPACE}]*[+-]?{DIGITS}[{ASCII_SPACE}]*$"
FLOAT_LITERAL = (
    f"(?i)^[{ASCII_SPACE}]*[+-]?"
    f"(({DIGITS}(\\.({DIGITS})?)?|\\.{DIGITS})(e[+-]?{DIGITS})?|inf|infinity|nan)"
    f"[{ASCII_SPACE}]*$"
)


@dataclass(frozen=True)
class TextPrimitives:
    matches: Callable[[Any, str], Any]
    strip: Callable[[Any, str], Any]
    replace: Callable[[Any, str, str], Any]
    byte_length: Callable[[Any], Any]
    char_length: Callable[[Any], Any]
    not_null: Callable[[Any], Any]
    try_int: Callable[[Any], Any]
    try_float: Callable[[Any], Any]


def is_ascii(primitives: TextPrimitives, text: Any) -> Any:
    return primitives.byte_length(text) == primitives.char_length(text)


def int_parses(primitives: TextPrimitives, text: Any) -> tuple[Any, Any]:
    limit = sys.get_int_max_str_digits() or sys.maxsize
    exact = is_ascii(primitives, text) & (primitives.char_length(text) <= limit)
    return primitives.matches(text, INT_LITERAL), exact


def float_parses(primitives: TextPrimitives, text: Any) -> tuple[Any, Any]:
    return primitives.matches(text, FLOAT_LITERAL), is_ascii(primitives, text)


def parse_int(primitives: TextPrimitives, text: Any) -> tuple[Any, Any]:
    return _parse(primitives, text, INT_LITERAL, primitives.try_int)


def parse_float(primitives: TextPrimitives, text: Any) -> tuple[Any, Any]:
    return _parse(primitives, text, FLOAT_LITERAL, primitives.try_float)


def _parse(
    primitives: TextPrimitives, text: Any, pattern: str, cast: Callable[[Any], Any]
) -> tuple[Any, Any]:
    cleaned = primitives.replace(primitives.strip(text, ASCII_SPACE), "_", "")
    value = cast(cleaned)
    literal = primitives.matches(text, pattern)
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
