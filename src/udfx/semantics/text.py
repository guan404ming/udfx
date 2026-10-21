from __future__ import annotations

import hashlib
import json
import operator
import os
import sys
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import cache, reduce
from pathlib import Path
from typing import Any, NamedTuple

from udfx.types import INT64_MAX

SIGMA = "\u03a3"
SIGMA_CONTEXTS = (f"A{SIGMA}{{}}b", f"A{{}}{SIGMA}", f"{{}}{SIGMA}", f"A{SIGMA}{{}}")

Recase = Callable[[Sequence[str]], Sequence[str]]
ANY_CHAR = "[\\s\\S]"
CLUSTER_JOINERS = (
    "[\\p{M}\\p{Cf}\\x{1F1E6}-\\x{1F1FF}\\x{1F3FB}-\\x{1F3FF}"
    "\\x{1100}-\\x{11FF}\\x{A960}-\\x{A97F}\\x{D7B0}-\\x{D7FF}]|\\r\\n"
)


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
def non_whitespace_pattern() -> str:
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


def case_guard(
    found: CaseDrift, contains_class: Callable[[str], Any], contains_sigma: Any
) -> Any | None:
    guards = [~contains_class(found.alone)] if found.alone else []
    if found.nearby:
        guards.append(~(contains_sigma & contains_class(found.nearby)))
    return reduce(operator.and_, guards) if guards else None


def case_drift(method: str, native: Recase, engine: str) -> CaseDrift:
    python = (
        f"python {sys.version_info[0]}.{sys.version_info[1]} unicode {unicodedata.unidata_version}"
    )
    key = f"{engine} {method} {python}"
    cached = read_cache(key)
    if cached is not None:
        return CaseDrift(**cached)
    reference: Callable[[str], str] = getattr(str, method)
    alone = drift(reference, native)
    contexts = (drift(reference, native, context) for context in SIGMA_CONTEXTS)
    nearby = frozenset[str]().union(*contexts) - alone
    widespread = len(nearby) > len(code_points()) // 2
    found = CaseDrift(
        char_class(alone) if alone else None,
        ANY_CHAR if widespread else char_class(nearby) if nearby else None,
    )
    write_cache(key, asdict(found))
    return found


def cache_path(key: str) -> Path:
    root = os.environ.get("UDFX_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "udfx" / f"case-drift-{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def read_cache(key: str) -> dict[str, str | None] | None:
    try:
        return json.loads(cache_path(key).read_text())
    except (OSError, ValueError):
        return None


def write_cache(key: str, value: dict[str, str | None]) -> None:
    path = cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    except OSError:
        return


def has_prefix_suffix_overlap(pattern: str) -> bool:
    return any(pattern[:size] == pattern[-size:] for size in range(1, len(pattern)))


@dataclass(frozen=True)
class SlicePrimitives:
    literal: Callable[[int], Any]
    when: Callable[[Any, Any, Any], Any]
    coalesce: Callable[[Any, Any], Any]
    clip: Callable[[Any, Any, Any], Any]


class SliceBounds(NamedTuple):
    start: Any
    length: Any


def slice_bounds(
    primitives: SlicePrimitives, text_length: Any, start: Any, stop: Any
) -> SliceBounds:
    zero = primitives.literal(0)
    maximum = primitives.literal(INT64_MAX)

    def clamp(bound: Any, default: Any) -> Any:
        shifted = bound + primitives.when(bound < zero, text_length, zero)
        return primitives.clip(primitives.coalesce(shifted, default), zero, text_length)

    begin = clamp(start, zero)
    end = clamp(stop, maximum)
    return SliceBounds(begin, primitives.clip(end - begin, zero, maximum))


@dataclass(frozen=True)
class SearchPrimitives:
    literal: Callable[[object], Any]
    when: Callable[[Any, Any, Any], Any]
    length: Callable[[Any], Any]
    reverse: Callable[[Any], Any]
    find: Callable[[Any, Any], Any]


def reverse_find(primitives: SearchPrimitives, text: Any, pattern: Any) -> Any:
    position = primitives.find(primitives.reverse(text), primitives.reverse(pattern))
    offset = primitives.length(text) - primitives.length(pattern) - position
    return primitives.when(position < primitives.literal(0), primitives.literal(-1), offset)


def count_occurrences(
    primitives: SearchPrimitives, text: Any, pattern: Any, occurrences: Any
) -> Any:
    empty = pattern == primitives.literal("")
    return primitives.when(empty, primitives.length(text) + primitives.literal(1), occurrences)
