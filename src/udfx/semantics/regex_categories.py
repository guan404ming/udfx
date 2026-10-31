import re
from collections.abc import Callable, Iterable
from functools import cache
from typing import Any

from udfx.semantics.text import code_points


def escaped(point: int) -> str:
    return f"\\x{{{point:X}}}"


def ascii_member(name: str, member: Any) -> Any:
    if name == "LITERAL":
        return member if member <= 127 else None
    if name == "RANGE":
        return (member[0], min(member[1], 127)) if member[0] <= 127 else None
    return member


def ranges(points: Iterable[int]) -> str:
    parts: list[str] = []
    start = previous = -1
    for point in points:
        if start >= 0 and point == previous + 1:
            previous = point
            continue
        if start >= 0:
            parts.append(escaped(start) + ("-" + escaped(previous) if start != previous else ""))
        start = previous = point
    if start >= 0:
        parts.append(escaped(start) + ("-" + escaped(previous) if start != previous else ""))
    return "".join(parts)


@cache
def category(name: str, ascii_only: bool = False) -> str:
    predicate = predicate_for(name)
    negate = "_NOT_" in name
    return ranges(
        ord(point)
        for point in code_points()
        if predicate(point) != negate and (point.isascii() or not ascii_only)
    )


def predicate_for(name: str) -> Callable[[str], bool]:
    return {
        "CATEGORY_DIGIT": str.isdecimal,
        "CATEGORY_SPACE": str.isspace,
        "CATEGORY_WORD": lambda point: point.isalnum() or point == "_",
    }[name.replace("_NOT_", "_")]


@cache
def ascii_characters(name: str, *, printable: bool = False) -> str:
    predicate = predicate_for(name)
    negate = "_NOT_" in name
    return "".join(
        re.escape(chr(point)) if 32 <= point < 127 else f"\\x{point:02X}"
        for point in range(32 if printable else 0, 127 if printable else 128)
        if predicate(chr(point)) != negate
    )


def corrected_category(name: str, pattern: str, matched: str) -> str:
    native = set(matched)
    predicate = predicate_for(name)
    expected = {point for point in code_points() if predicate(point)}
    added = ranges(sorted(map(ord, expected - native)))
    removed = ranges(sorted(map(ord, native - expected)))
    correction = "&&[^" + removed + "]" if removed else ""
    corrected = "[" + pattern + added + correction + "]"
    return "[[" + ascii_characters(name) + r"][[\x{80}-\x{10ffff}]&&" + corrected + "]]"
