from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from udfx.semantics.sequences import SequencePrimitives


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


def split_limited(
    primitives: SplitPrimitives, text: Any, separator: Any, maximum: Any, partitioned: bool = False
) -> Any:
    if partitioned:
        return split_partition(primitives, text, separator, maximum)
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
) -> Any:
    if partitioned:
        return split_partition(primitives, text, separator, maximum, right=True)
    reversed_text = primitives.reverse(text)
    reversed_separator = primitives.reverse(separator)
    pieces = (
        primitives.split(reversed_text, reversed_separator)
        if maximum is None
        else split_limited(primitives, reversed_text, reversed_separator, maximum)
    )
    return primitives.restore(pieces)
