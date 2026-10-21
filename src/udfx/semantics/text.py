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

from udfx.ir import Call, Lit, Node
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


@cache
def word_prefix_pattern() -> str:
    return char_class(whitespace()) + "*" + non_whitespace_pattern()


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
    maximum: int


class SliceBounds(NamedTuple):
    start: Any
    length: Any


def slice_bounds(
    primitives: SlicePrimitives, text_length: Any, start: Any, stop: Any, reverse: Any = None
) -> SliceBounds:
    zero = primitives.literal(0)
    maximum = primitives.literal(primitives.maximum)

    if isinstance(start, int) and start >= primitives.maximum:
        return SliceBounds(zero, zero)
    if (start is None or isinstance(start, int)) and (stop is None or isinstance(stop, int)):
        begin = 0 if start is None else start
        end = primitives.maximum if stop is None else stop
        if begin >= 0 and end >= 0:
            return SliceBounds(
                primitives.literal(min(begin, primitives.maximum - 1)),
                primitives.literal(min(max(end - begin, 0), primitives.maximum)),
            )

    lower = zero if reverse is None else primitives.when(reverse, primitives.literal(-1), zero)
    upper = text_length + lower

    def clamp(bound: Any, default: Any) -> Any:
        if bound is None:
            return default
        if isinstance(bound, int):
            if bound >= 0:
                return primitives.literal(min(bound, primitives.maximum))
            return primitives.clip(
                text_length + primitives.literal(max(bound, -INT64_MAX)), zero, text_length
            )
        shifted = bound + primitives.when(bound < zero, text_length, zero)
        bounded = primitives.clip(primitives.coalesce(shifted, default), lower, upper)
        return (
            bounded
            if reverse is None
            else primitives.when(reverse, text_length - primitives.literal(1) - bounded, bounded)
        )

    begin = clamp(start, zero if reverse is None else primitives.when(reverse, upper, zero))
    end = clamp(stop, maximum if reverse is None else primitives.when(reverse, lower, maximum))
    return SliceBounds(begin, primitives.clip(end - begin, zero, maximum))


@dataclass(frozen=True)
class SequencePrimitives:
    bounds: SlicePrimitives
    length: Callable[[Any], Any]
    slice: Callable[[Any, Any, Any], Any]
    reverse: Callable[[Any], Any]
    stride: Callable[[Any, Any], Any]


def slice_sequence(
    primitives: SequencePrimitives, container: Any, start: Any, stop: Any, step: Any = None
) -> Any:
    size = primitives.length(container)
    bounds = primitives.bounds
    if step is None:
        begin, length = slice_bounds(bounds, size, start, stop)
        return primitives.slice(container, begin, length)
    zero, one = bounds.literal(0), bounds.literal(1)
    step = bounds.coalesce(step, one)
    reverse = step < zero
    begin, length = slice_bounds(bounds, size, start, stop, reverse)
    ordered = bounds.when(reverse, primitives.reverse(container), container)
    sliced = primitives.slice(ordered, begin, length)
    maximum = bounds.clip(size, one, bounds.literal(INT64_MAX))
    stride = bounds.when(
        reverse,
        zero - bounds.clip(step, zero - maximum, bounds.literal(-1)),
        bounds.clip(step, one, maximum),
    )
    return primitives.stride(sliced, stride)


@dataclass(frozen=True)
class SplitPrimitives:
    sequence: SequencePrimitives
    split: Callable[[Any, Any], Any]
    join: Callable[[Any, Any], Any]
    append: Callable[[Any, Any], Any]
    length: Callable[[Any], Any]
    suffix: Callable[[Any, Any], Any]
    word_prefix: Callable[[Any, Any], Any]
    lstrip: Callable[[Any], Any]
    is_null: Callable[[Any], Any]
    reverse: Callable[[Any], Any]
    restore: Callable[[Any], Any]


def split_limited(primitives: SplitPrimitives, text: Any, separator: Any, maximum: Any) -> Any:
    sequence = primitives.sequence
    bounds = sequence.bounds
    zero = bounds.literal(0)
    pieces = primitives.split(text, separator)
    size = sequence.length(pieces)
    count = bounds.clip(maximum, zero, size)
    prefix = sequence.slice(pieces, zero, count)
    offset = primitives.length(primitives.join(separator, prefix)) + primitives.length(separator)
    tail = primitives.suffix(text, bounds.when(count == zero, zero, offset))
    word_offset = primitives.length(primitives.word_prefix(text, count))
    word_tail = primitives.lstrip(primitives.suffix(text, word_offset))
    remainder = bounds.when(primitives.is_null(separator), word_tail, tail)
    limited = primitives.append(prefix, remainder)
    return bounds.when((maximum < zero) | (maximum >= size), pieces, limited)


def split_right(primitives: SplitPrimitives, text: Any, separator: Any, maximum: Any = None) -> Any:
    reversed_text = primitives.reverse(text)
    reversed_separator = primitives.reverse(separator)
    pieces = (
        primitives.split(reversed_text, reversed_separator)
        if maximum is None
        else split_limited(primitives, reversed_text, reversed_separator, maximum)
    )
    return primitives.restore(pieces)


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


def maximum_length(node: Node) -> int | None:
    if isinstance(node, Lit) and isinstance(node.value, str):
        return len(node.value)
    if not isinstance(node, Call):
        return None
    if node.op.name in {".strip", ".lstrip", ".rstrip"}:
        return maximum_length(node.arguments[0])
    if node.op.name != "slice":
        return None
    text, start, stop = node.arguments
    bounds = [maximum_length(text)]
    if isinstance(start, Lit) and isinstance(start.value, int) and start.value < 0:
        bounds.append(-start.value)
    if isinstance(stop, Lit) and isinstance(stop.value, int) and stop.value >= 0:
        begin = start.value if isinstance(start, Lit) and isinstance(start.value, int) else 0
        bounds.append(max(stop.value - max(begin, 0), 0))
    known = [bound for bound in bounds if bound is not None]
    return min(known) if known else None
