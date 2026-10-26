import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from udfx.semantics.text import string_predicate

ASCII_PREDICATES = {
    "isdigit": r"[0-9]+",
    "isdecimal": r"[0-9]+",
    "isnumeric": r"[0-9]+",
    "isalpha": r"[A-Za-z]+",
    "isalnum": r"[A-Za-z0-9]+",
    "isspace": r"[\t\n\x0b\x0c\r\x1c-\x1f ]+",
    "islower": r"[^A-Z]*[a-z][^A-Z]*",
    "isupper": r"[^a-z]*[A-Z][^a-z]*",
}


@dataclass(frozen=True)
class AsciiPrimitives:
    is_ascii: Callable[[Any], Any]
    when: Callable[[Any, Any, Any], Any]
    literal: Callable[[object], Any]


def ascii_predicate(
    primitives: AsciiPrimitives,
    matches: Callable[[Any, str], Any],
    value: Any,
    method: str,
    known: bool | None = None,
) -> Any:
    ascii = primitives.is_ascii(value)
    if method == "isascii":
        return ascii
    fast = (
        primitives.literal(known)
        if known is not None
        else matches(value, "[A-Z]") & ~matches(value, r"(?:\A|[^A-Za-z])[a-z]|[A-Za-z][A-Z]")
        if method == "istitle"
        else matches(value, rf"\A{ASCII_PREDICATES[method]}\z")
    )
    return primitives.when(
        ascii,
        fast,
        string_predicate(matches, value, method),
    )


def ascii_free(pattern: str) -> bool:
    points = re.findall(r"\\x\{([0-9A-F]+)\}", pattern)
    return bool(points) and min(int(point, 16) for point in points) >= 0x80


def unicode_match(
    primitives: AsciiPrimitives, matches: Callable[[Any, str], Any], value: Any, pattern: str
) -> Any:
    found = matches(value, pattern)
    if not ascii_free(pattern):
        return found
    return primitives.when(primitives.is_ascii(value), primitives.literal(False), found)
