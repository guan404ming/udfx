from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from udfx.semantics.sequences import SequencePrimitives
from udfx.semantics.text import NATIVE_PADDING_LIMIT


@dataclass(frozen=True)
class PaddingPrimitives:
    sequence: SequencePrimitives
    literal: Callable[[object], Any]
    concat: Callable[..., Any]
    repeat: Callable[[Any, Any], Any]
    half: Callable[[Any], Any]


def pad_text(
    primitives: PaddingPrimitives,
    value: Any,
    width: Any,
    fill: Any,
    method: str,
    *,
    maximum: int | None = None,
) -> tuple[Any, Any]:
    sequence = primitives.sequence
    literal = primitives.literal
    when = sequence.bounds.when
    zero = literal(0)
    limit = literal(NATIVE_PADDING_LIMIT)
    exact = None if maximum is not None and maximum <= NATIVE_PADDING_LIMIT else width <= limit
    positive = width > zero
    width = when(positive if exact is None else positive & exact, width, zero)
    padding = sequence.bounds.clip(width - sequence.length(value), zero, limit)
    if method == "zfill":
        zeros = primitives.repeat(literal("0"), padding)
        head = sequence.head(value, 1)
        signed = (head == literal("+")) | (head == literal("-"))
        return when(
            signed,
            primitives.concat(head, zeros, sequence.drop_head(value, 1)),
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
