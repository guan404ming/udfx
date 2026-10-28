from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from udfx.semantics.json_lowering import JsonPrimitives


def within_limits(primitives: JsonPrimitives, value: Any, structure: Any) -> Any:
    text = primitives.text
    length = primitives.byte_length(value)
    depth = text.literal(sys.getrecursionlimit() // 2)
    without_openings = text.replace(text.replace(value, "[", ""), "{", "")
    upper = length - primitives.byte_length(without_openings)
    openings = primitives.replace(structure, r"[^\[{]", "")
    exact = text.when(upper < depth, text.literal(True), text.char_length(openings) < depth)
    digits = sys.get_int_max_str_digits()
    if digits:
        exact = exact & text.when(
            length <= text.literal(digits),
            text.literal(True),
            text.when(
                text.char_length(value) <= text.literal(digits),
                text.literal(True),
                primitives.longest_digits(structure) <= text.literal(digits),
            ),
        )
    return exact
