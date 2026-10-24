from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from udfx.semantics.literals import integer_characters
from udfx.semantics.text import character_class


@dataclass(frozen=True)
class CalendarPrimitives:
    literal: Callable[[object], Any]
    select: Callable[[Any, Any, Any], Any]
    matches: Callable[[Any, str], Any]
    translate: Callable[[Any, str, str], Any]
    replace: Callable[[Any, str, str], Any]
    parse: Callable[[Any], Any]
    year: Callable[[Any], Any]


def parse_calendar(primitives: CalendarPrimitives, text: Any, format: str) -> Any | None:
    if format not in {"%Y", "%Y-%m", "%Y-%m-%d"}:
        return None
    digits = character_class("isdecimal")
    pattern = rf"{digits}{{4}}"
    ascii_pattern = r"[0-9]{4}"
    if "%m" in format:
        pattern += r"-(1[0-2]|0[1-9]|[1-9])"
        ascii_pattern += r"-[0-9]{1,2}"
    if "%d" in format:
        pattern += rf"-(3[01]|[12]{digits}|0[1-9]|[1-9]| [1-9])"
        ascii_pattern += r"-[0-9]{1,2}"
    original, replacement = integer_characters()
    normalized = primitives.replace(primitives.translate(text, original, replacement), "- ", "-")
    source = primitives.select(primitives.matches(text, rf"\A{ascii_pattern}\z"), text, normalized)
    parsed = primitives.parse(source)
    year = primitives.year(parsed)
    valid = (
        primitives.matches(text, rf"\A{pattern}\z")
        & (year >= primitives.literal(1))
        & (year <= primitives.literal(9999))
    )
    return primitives.select(valid, parsed, primitives.literal(None))
