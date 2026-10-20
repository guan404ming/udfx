from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import cache


@cache
def code_points() -> tuple[str, ...]:
    return tuple(chr(point) for point in range(0x110000) if not 0xD800 <= point <= 0xDFFF)


@cache
def whitespace() -> str:
    return "".join(point for point in code_points() if point.isspace())


def char_class(chars: Iterable[str], negate: bool = False) -> str:
    body = "".join(f"\\x{{{ord(char):X}}}" for char in chars)
    return f"[{'^' if negate else ''}{body}]"


@cache
def word() -> str:
    return char_class(whitespace(), negate=True) + "+"


def drift(
    reference: Callable[[str], str], native: Callable[[Sequence[str]], Sequence[str]]
) -> tuple[str, ...]:
    points = code_points()
    mapped = native(points)
    return tuple(
        point for point, result in zip(points, mapped, strict=True) if reference(point) != result
    )
