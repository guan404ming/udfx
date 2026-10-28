from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from udfx.semantics.text import char_class
from udfx.semantics.text_slicing import mixed_slice_pattern
from udfx.types import INT64_MAX, Type


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
    drop_head: Callable[[Any, int], Any]
    drop_tail: Callable[[Any, int], Any]
    reverse: Callable[[Any], Any]
    stride: Callable[[Any, int], Any]
    bind: Callable[[Any, Callable[[Any], Any], Type], Any]
    extract: Callable[[Any, str, int], Any] | None = None


def take_prefix(primitives: SequencePrimitives, container: Any, count: Any) -> Any:
    bounds = primitives.bounds
    zero = bounds.literal(0)
    size = bounds.clip(count, zero, bounds.literal(bounds.maximum))
    return primitives.slice(container, zero, size)


def slice_forward(
    primitives: SequencePrimitives,
    container: Any,
    start: Any,
    stop: Any,
    result: Type,
    *,
    length: Any = None,
) -> Any:
    bounds = primitives.bounds

    def head(value: Any, size: int) -> Any:
        return primitives.head(value, size) if size >= 0 else primitives.drop_tail(value, -size)

    def tail(value: Any, size: int) -> Any:
        return primitives.tail(value, size) if size > 0 else primitives.drop_head(value, -size)

    def bounded(value: Any) -> Any:
        size = primitives.length(value) if length is None else length
        begin, count = slice_bounds(bounds, size, start, stop)
        return primitives.slice(value, begin, count)

    if (start is None or isinstance(start, int)) and (stop is None or isinstance(stop, int)):
        start = max(min(start or 0, bounds.maximum), -bounds.maximum)
        if start == bounds.maximum or (stop is not None and stop <= -bounds.maximum):
            return primitives.head(container, 0)
        if stop is None:
            return tail(container, -start) if start else container
        stop = max(min(stop, bounds.maximum), -bounds.maximum)
        if start == 0 or stop == 0:
            return head(container, stop)
        if stop < 0:
            if start > 0:
                return tail(head(container, stop), -start)
            return head(tail(container, -start), stop)
        if start < 0:
            if primitives.extract is not None:
                pattern = mixed_slice_pattern(start, stop)
                if pattern is not None:
                    return primitives.extract(container, pattern, 2)
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
    dynamic = any(bound is not None and not isinstance(bound, int) for bound in (start, stop))
    length = primitives.length(container) if step < 0 and dynamic else None
    if step < 0:
        container = primitives.reverse(container)
        start = reverse_bound(primitives.bounds, start)
        stop = reverse_bound(primitives.bounds, stop)
    sliced = slice_forward(primitives, container, start, stop, result, length=length)
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


def reverse_exact(step: int | None, exact: Callable[[], Any]) -> Any:
    return exact() if step is not None and step < 0 else None


def stride_pattern(step: int) -> str | None:
    return f"(?s)(.).{{0,{step - 1}}}" if step <= 64 else None


def token_stride_pattern(separator: str, step: int) -> str | None:
    if step > 64:
        return None
    boundary = char_class(separator)
    token = char_class(separator, negate=True) + "*"
    return f"(?:{boundary}{token}){{{step - 1}}}{boundary}({token})"
