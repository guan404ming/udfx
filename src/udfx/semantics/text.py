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
from udfx.ops.registry import apply
from udfx.types import INT64_MAX, Type

SIGMA = "\u03a3"
SIGMA_CONTEXTS = (f"A{SIGMA}{{}}b", f"A{{}}{SIGMA}", f"{{}}{SIGMA}", f"A{SIGMA}{{}}")

Recase = Callable[[Sequence[str]], Sequence[str]]
ANY_CHAR = "[\\s\\S]"
CLUSTER_JOINERS = (
    "[\\p{M}\\p{Cf}\\x{1F1E6}-\\x{1F1FF}\\x{1F3FB}-\\x{1F3FF}"
    "\\x{1100}-\\x{11FF}\\x{A960}-\\x{A97F}\\x{D7B0}-\\x{D7FF}]|\\r\\n"
)
PREDICATES = (
    "isdigit",
    "isalpha",
    "isalnum",
    "isnumeric",
    "isdecimal",
    "isspace",
    "islower",
    "isupper",
    "istitle",
)
CASE_METHODS = ("lower", "upper", "title", "capitalize", "swapcase", "casefold")
PADDING_METHODS = ("zfill", "center", "ljust", "rjust")
NATIVE_PADDING_LIMIT = 2**20


@cache
def code_points() -> tuple[str, ...]:
    return tuple(chr(point) for point in range(0x110000) if not 0xD800 <= point <= 0xDFFF)


@cache
def whitespace() -> str:
    return "".join(point for point in code_points() if point.isspace())


def char_class(chars: Iterable[str], negate: bool = False) -> str:
    ranges: list[str] = []
    points = sorted({ord(char) for char in chars})
    start = end = points[0] if points else 0
    for point in points[1:]:
        if point == end + 1:
            end = point
        else:
            ranges.append(f"\\x{{{start:X}}}" + (f"-\\x{{{end:X}}}" if start != end else ""))
            start = end = point
    if points:
        ranges.append(f"\\x{{{start:X}}}" + (f"-\\x{{{end:X}}}" if start != end else ""))
    return f"[{'^' if negate else ''}{''.join(ranges)}]"


@cache
def character_class(method: str) -> str:
    predicate: Callable[[str], bool] = getattr(str, method)
    return char_class(point for point in code_points() if predicate(point))


@cache
def cased_characters() -> str:
    return "".join(
        point for point in code_points() if point.islower() or point.isupper() or point.istitle()
    )


@cache
def predicate_patterns(method: str) -> tuple[str, str | None]:
    if method not in {"islower", "isupper", "istitle"}:
        return rf"\A{character_class(method)}+\z", None
    uncased = char_class(cased_characters(), negate=True)
    if method == "istitle":
        title = character_class("istitle")
        lowercase_start = rf"(?:\A|{uncased}){character_class('islower')}"
        uppercase_continuation = char_class(cased_characters()) + title
        return title, f"{lowercase_start}|{uppercase_continuation}"
    forbidden = char_class(
        (point for point in cased_characters() if not getattr(point, method)()),
        negate=True,
    )
    return rf"\A{uncased}*{character_class(method)}{forbidden}*\z", None


def string_predicate(matches: Callable[[Any, str], Any], value: Any, method: str) -> Any:
    required, forbidden = predicate_patterns(method)
    found = matches(value, required)
    return found if forbidden is None else found & ~matches(value, forbidden)


@cache
def non_whitespace_pattern() -> str:
    return char_class(whitespace(), negate=True) + "+"


@cache
def word_prefix_pattern() -> str:
    return char_class(whitespace()) + "*" + non_whitespace_pattern()


def strip_pattern(characters: str, method: str) -> str:
    pattern = char_class(characters) + "+"
    if method == "lstrip":
        return "^" + pattern
    if method == "rstrip":
        return pattern + "$"
    return "^" + pattern + "|" + pattern + "$"


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
    if method in {"title", "capitalize"}:
        alone |= frozenset[str]().union(
            *(drift(reference, native, context) for context in ("A{}b", "a{}B", "{}b", "a{}"))
        )
    contexts = (drift(reference, native, context) for context in SIGMA_CONTEXTS)
    nearby = frozenset[str]().union(*contexts) - alone
    widespread = len(nearby) > len(code_points()) // 2
    found = CaseDrift(
        char_class(alone) if alone else None,
        ANY_CHAR if widespread else char_class(nearby) if nearby else None,
    )
    write_cache(key, asdict(found))
    return found


@dataclass(frozen=True)
class CasePrimitives:
    lower: Callable[[Any], Any]
    upper: Callable[[Any], Any]
    title: Callable[[Any], Any]
    head: Callable[[Any], Any]
    tail: Callable[[Any], Any]
    concat: Callable[..., Any]
    when: Callable[[Any, Any, Any], Any]
    matches: Callable[[Any, str], Any]
    characters: Callable[[Any, Callable[[Any], Any]], Any]


def capitalized(primitives: CasePrimitives, value: Any) -> Any:
    return primitives.concat(
        primitives.upper(primitives.head(value)), primitives.lower(primitives.tail(value))
    )


def title_words(
    primitives: CasePrimitives,
    value: Any,
    transform: Callable[[Any, str, Callable[[Any], Any]], Any],
) -> Any:
    pattern = char_class(cased_characters()) + "+|" + char_class(cased_characters(), True) + "+"
    return transform(value, pattern, lambda word: capitalized(primitives, word))


def case_expression(primitives: CasePrimitives, value: Any, method: str) -> Any:
    if method in {"lower", "casefold"}:
        return primitives.lower(value)
    if method == "upper":
        return primitives.upper(value)
    if method == "title":
        return primitives.title(value)
    if method == "capitalize":
        return capitalized(primitives, value)
    return primitives.characters(
        value,
        lambda character: primitives.when(
            primitives.matches(character, character_class("islower")),
            primitives.upper(character),
            primitives.lower(character),
        ),
    )


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


def simplify_text(node: Call) -> Call:
    if node.op.name in {".lower", ".upper"}:
        argument = node.arguments[0]
        if (
            isinstance(argument, Call)
            and argument.op.name in {".strip", ".lstrip", ".rstrip"}
            and len(argument.arguments) == 1
        ):
            return apply(argument.op.name, apply(node.op.name, argument.arguments[0]))
    if node.op.name != "slice":
        return node
    text, start, stop, *steps = node.arguments
    if steps and not (isinstance(steps[0], Lit) and steps[0].value in (None, 1)):
        return node
    if not isinstance(stop, Lit) or stop.value is not None:
        return node
    if not isinstance(start, Call) or start.op.name != "__add__":
        return node
    search, offset = start.arguments
    if not isinstance(search, Call) or search.op.name != ".rfind":
        return node
    source, pattern = search.arguments
    if (
        source == text
        and isinstance(pattern, Lit)
        and isinstance(pattern.value, str)
        and len(pattern.value) == 1
        and isinstance(offset, Lit)
        and offset.value == 1
    ):
        return apply("last_split", text, pattern)
    return node


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
    primitives: SlicePrimitives, text_length: Any, start: Any, stop: Any
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
        return primitives.clip(primitives.coalesce(shifted, default), zero, text_length)

    begin = clamp(start, zero)
    end = clamp(stop, maximum)
    return SliceBounds(begin, primitives.clip(end - begin, zero, maximum))


@dataclass(frozen=True)
class SequencePrimitives:
    bounds: SlicePrimitives
    length: Callable[[Any], Any]
    slice: Callable[[Any, Any, Any], Any]
    head: Callable[[Any, int], Any]
    tail: Callable[[Any, int], Any]
    reverse: Callable[[Any], Any]
    stride: Callable[[Any, int], Any]
    bind: Callable[[Any, Callable[[Any], Any], Type], Any]


@dataclass(frozen=True)
class PaddingPrimitives:
    sequence: SequencePrimitives
    literal: Callable[[object], Any]
    concat: Callable[..., Any]
    repeat: Callable[[Any, Any], Any]
    half: Callable[[Any], Any]


def pad_text(
    primitives: PaddingPrimitives, value: Any, width: Any, fill: Any, method: str
) -> tuple[Any, Any]:
    sequence = primitives.sequence
    literal = primitives.literal
    when = sequence.bounds.when
    zero = literal(0)
    limit = literal(NATIVE_PADDING_LIMIT)
    exact = width <= limit
    width = when((width > zero) & exact, width, zero)
    padding = sequence.bounds.clip(width - sequence.length(value), zero, limit)
    if method == "zfill":
        zeros = primitives.repeat(literal("0"), padding)
        head = sequence.head(value, 1)
        signed = (head == literal("+")) | (head == literal("-"))
        return when(
            signed,
            primitives.concat(head, zeros, sequence.tail(value, -1)),
            primitives.concat(zeros, value),
        ), exact
    fill = sequence.head(fill, 1)
    if method == "center":
        left = primitives.half(padding) + padding % literal(2) * (width % literal(2))
        right = padding - left
    else:
        left, right = (padding, zero) if method == "rjust" else (zero, padding)
    return primitives.concat(
        primitives.repeat(fill, left), value, primitives.repeat(fill, right)
    ), exact


def slice_forward(
    primitives: SequencePrimitives, container: Any, start: Any, stop: Any, result: Type
) -> Any:
    bounds = primitives.bounds

    def bounded(value: Any) -> Any:
        begin, length = slice_bounds(bounds, primitives.length(value), start, stop)
        return primitives.slice(value, begin, length)

    if (start is None or isinstance(start, int)) and (stop is None or isinstance(stop, int)):
        start = max(min(start or 0, bounds.maximum), -bounds.maximum)
        if start == bounds.maximum or (stop is not None and stop <= -bounds.maximum):
            return primitives.head(container, 0)
        if stop is None:
            return primitives.tail(container, -start) if start else container
        stop = max(min(stop, bounds.maximum), -bounds.maximum)
        if start == 0 or stop == 0:
            return primitives.head(container, stop)
        if stop < 0:
            if start > 0:
                return primitives.tail(primitives.head(container, stop), -start)
            return primitives.head(primitives.tail(container, -start), stop)
        if start < 0:
            return primitives.bind(container, bounded, result)
    return bounded(container)


def slice_sequence(
    primitives: SequencePrimitives,
    container: Any,
    start: Any,
    stop: Any,
    step: int | None = None,
    *,
    result: Type,
) -> Any:
    step = step or 1
    if step < 0:
        container = primitives.reverse(container)
        start = reverse_bound(primitives.bounds, start)
        stop = reverse_bound(primitives.bounds, stop)
    sliced = slice_forward(primitives, container, start, stop, result)
    if abs(step) == 1:
        return sliced
    return primitives.stride(sliced, min(abs(step), primitives.bounds.maximum))


def reverse_bound(primitives: SlicePrimitives, bound: Any) -> Any:
    if bound is None:
        return None
    if isinstance(bound, int):
        return -1 - bound
    return primitives.literal(-1) - bound


@dataclass(frozen=True)
class SplitPrimitives:
    sequence: SequencePrimitives
    bind: Callable[[Sequence[Any], Callable[..., Any]], Any]
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

    def limited(text: Any, separator: Any, maximum: Any, pieces: Any) -> Any:
        size = sequence.length(pieces)
        count = bounds.clip(maximum, zero, size)
        prefix = sequence.slice(pieces, zero, count)
        offset = primitives.length(primitives.join(separator, prefix)) + primitives.length(
            separator
        )
        tail = primitives.suffix(text, bounds.when(count == zero, zero, offset))
        word_offset = primitives.length(primitives.word_prefix(text, count))
        word_tail = primitives.lstrip(primitives.suffix(text, word_offset))
        remainder = bounds.when(primitives.is_null(separator), word_tail, tail)
        return bounds.when(
            (maximum < zero) | (maximum >= size), pieces, primitives.append(prefix, remainder)
        )

    return primitives.bind(
        (text, separator, maximum),
        lambda text, separator, maximum: primitives.bind(
            (text, separator, maximum, primitives.split(text, separator)), limited
        ),
    )


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
    text, start, stop, *steps = node.arguments
    if steps and (not isinstance(steps[0], Lit) or steps[0].value not in (None, 1)):
        return maximum_length(text)
    bounds = [maximum_length(text)]
    if isinstance(start, Lit) and isinstance(start.value, int) and start.value < 0:
        bounds.append(-start.value)
    if isinstance(stop, Lit) and isinstance(stop.value, int) and stop.value >= 0:
        begin = start.value if isinstance(start, Lit) and isinstance(start.value, int) else 0
        bounds.append(max(stop.value - max(begin, 0), 0))
    known = [bound for bound in bounds if bound is not None]
    return min(known) if known else None
