from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cache

SIGMA = "\u03a3"
SIGMA_CONTEXTS = (f"A{SIGMA}{{}}b", f"A{{}}{SIGMA}", f"{{}}{SIGMA}", f"A{SIGMA}{{}}")

Recase = Callable[[Sequence[str]], Sequence[str]]


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


def drift(reference: Callable[[str], str], native: Recase, context: str = "{}") -> frozenset[str]:
    points = code_points()
    samples = [context.format(point) for point in points]
    mapped = native(samples)
    return frozenset(
        point
        for point, sample, result in zip(points, samples, mapped, strict=True)
        if reference(sample) != result
    )


@dataclass(frozen=True)
class CaseDrift:
    alone: str | None
    nearby: str | None


def case_drift(method: str, native: Recase) -> CaseDrift:
    reference: Callable[[str], str] = getattr(str, method)
    alone = drift(reference, native)
    contexts = (drift(reference, native, context) for context in SIGMA_CONTEXTS)
    nearby = frozenset[str]().union(*contexts) - alone
    return CaseDrift(char_class(alone) if alone else None, char_class(nearby) if nearby else None)
