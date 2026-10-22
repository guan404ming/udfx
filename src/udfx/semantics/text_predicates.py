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


def ascii_predicate(
    primitives: AsciiPrimitives, matches: Callable[[Any, str], Any], value: Any, method: str
) -> Any:
    ascii = primitives.is_ascii(value)
    if method == "isascii":
        return ascii
    fast = (
        matches(value, "[A-Z]") & ~matches(value, r"(?:\A|[^A-Za-z])[a-z]|[A-Za-z][A-Z]")
        if method == "istitle"
        else matches(value, rf"\A{ASCII_PREDICATES[method]}\z")
    )
    return primitives.when(
        ascii,
        fast,
        string_predicate(matches, value, method),
    )
