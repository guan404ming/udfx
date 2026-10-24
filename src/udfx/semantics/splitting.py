from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from udfx.semantics.sequences import SequencePrimitives


@dataclass(frozen=True)
class SplitPrimitives:
    sequence: SequencePrimitives
    bind: Callable[[Sequence[Any], Callable[..., Any]], Any]
    split: Callable[[Any, Any], Any]
    words: Callable[[Any], Any]
    join: Callable[[Any, Any], Any]
    append: Callable[[Any, Any], Any]
    length: Callable[[Any], Any]
    suffix: Callable[[Any, Any], Any]
    word_prefix: Callable[[Any, Any], Any]
    lstrip: Callable[[Any], Any]
    is_null: Callable[[Any], Any]
    reverse: Callable[[Any], Any]
    restore: Callable[[Any], Any]
    limited: Callable[[Any, Any, Any], Any] | None = None


def split_limited(
    primitives: SplitPrimitives,
    text: Any,
    separator: Any,
    maximum: Any,
    partitioned: bool = False,
    *,
    optional: bool | None = None,
) -> Any:
    bounds = primitives.sequence.bounds
    zero = bounds.literal(0)
    if partitioned:
        return split_partition(primitives, text, separator, maximum)
    if optional is True:
        return split_word_limited(primitives, text, maximum)
    if primitives.limited is not None:
        limit = bounds.when(
            (maximum < zero) | (maximum >= bounds.literal(bounds.maximum)),
            bounds.literal(-1),
            maximum + bounds.literal(1),
        )
        result = primitives.limited(text, separator, limit)
        return (
            result
            if optional is False
            else bounds.when(
                primitives.is_null(separator),
                split_word_limited(primitives, text, maximum),
                result,
            )
        )
    sequence = primitives.sequence

    def limited(text: Any, separator: Any, maximum: Any, pieces: Any) -> Any:
        size = sequence.length(pieces)
        count = bounds.clip(maximum, zero, size)
        prefix = sequence.slice(pieces, zero, count)
        offset = primitives.length(primitives.join(separator, prefix)) + primitives.length(
            separator
        )
        tail = primitives.suffix(text, bounds.when(count == zero, zero, offset))
        if optional is None:
            word_offset = primitives.length(primitives.word_prefix(text, count))
            word_tail = primitives.lstrip(primitives.suffix(text, word_offset))
            tail = bounds.when(primitives.is_null(separator), word_tail, tail)
        return bounds.when(
            (maximum < zero) | (maximum >= size), pieces, primitives.append(prefix, tail)
        )

    return primitives.bind(
        (text, separator, maximum),
        lambda text, separator, maximum: primitives.bind(
            (text, separator, maximum, primitives.split(text, separator)), limited
        ),
    )


def split_word_limited(primitives: SplitPrimitives, text: Any, maximum: Any) -> Any:
    sequence = primitives.sequence
    bounds = sequence.bounds
    zero = bounds.literal(0)

    def limited(text: Any, maximum: Any, pieces: Any) -> Any:
        size = sequence.length(pieces)
        count = bounds.clip(maximum, zero, size)
        prefix = sequence.slice(pieces, zero, count)
        offset = primitives.length(primitives.word_prefix(text, count))
        remainder = primitives.lstrip(primitives.suffix(text, offset))
        return bounds.when(
            (maximum < zero) | (maximum >= size), pieces, primitives.append(prefix, remainder)
        )

    return primitives.bind(
        (text, maximum),
        lambda text, maximum: primitives.bind((text, maximum, primitives.words(text)), limited),
    )


def split_partition(
    primitives: SplitPrimitives, text: Any, separator: Any, maximum: Any, right: bool = False
) -> Any:
    sequence = primitives.sequence
    bounds = sequence.bounds
    pieces = primitives.split(text, separator)
    if maximum is None:
        return pieces
    zero = bounds.literal(0)
    size = sequence.length(pieces)
    count = bounds.clip(maximum, zero, size - bounds.literal(1))
    boundary = size - count if right else count
    prefix = sequence.slice(pieces, zero, boundary)
    suffix = sequence.slice(pieces, boundary, size - boundary)
    if right:
        result = sequence.reverse(
            primitives.append(sequence.reverse(suffix), primitives.join(separator, prefix))
        )
    else:
        result = primitives.append(prefix, primitives.join(separator, suffix))
    return bounds.when(maximum < zero, pieces, result)


def split_right(
    primitives: SplitPrimitives,
    text: Any,
    separator: Any,
    maximum: Any = None,
    partitioned: bool = False,
    *,
    optional: bool | None = None,
) -> Any:
    if partitioned:
        return split_partition(primitives, text, separator, maximum, right=True)
    reversed_text = primitives.reverse(text)
    reversed_separator = primitives.reverse(separator)
    pieces = (
        primitives.split(reversed_text, reversed_separator)
        if maximum is None
        else split_limited(
            primitives, reversed_text, reversed_separator, maximum, optional=optional
        )
    )
    restored = primitives.restore(pieces)
    if optional is True and maximum is not None:
        sequence = primitives.sequence
        words = primitives.words(text)
        return sequence.bounds.when(
            (maximum < sequence.bounds.literal(0)) | (maximum >= primitives.length(text)),
            words,
            restored,
        )
    return restored


@dataclass(frozen=True)
class RightSplitPrimitives:
    split: SplitPrimitives
    head: Callable[[Any, Any], Any]
    contains: Callable[[Any, Any], Any]
    tail: Callable[[Any, Any, Any], Any]
    singleton: Callable[[Any], Any]
    concatenate: Callable[[Any, Any], Any]


def split_right_native(
    primitives: RightSplitPrimitives,
    text: Any,
    separator: Any,
    maximum: Any,
    fallback: Any,
) -> Any:
    split = primitives.split
    bounds = split.sequence.bounds
    zero = bounds.literal(0)
    unique_head = ~primitives.contains(
        split.suffix(separator, bounds.literal(1)), primitives.head(separator, bounds.literal(1))
    )
    partitioned = (split.length(separator) > zero) & unique_head
    result = split.split(text, separator)
    if maximum is not None:
        count = bounds.clip(maximum, zero, bounds.literal(bounds.maximum))
        tail = primitives.tail(text, separator, count)
        length = split.length(text) - split.length(tail) - split.length(separator)
        prefix = primitives.head(text, length)
        limited = primitives.concatenate(primitives.singleton(prefix), split.split(tail, separator))
        result = bounds.when(
            maximum == zero,
            primitives.singleton(text),
            bounds.when((maximum < zero) | (length < zero), result, limited),
        )
    return bounds.when(
        ~primitives.contains(text, separator),
        primitives.singleton(text),
        bounds.when(partitioned, result, fallback),
    )
