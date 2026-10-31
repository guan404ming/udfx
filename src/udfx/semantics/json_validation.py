from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from udfx.semantics.json_limits import within_limits
from udfx.semantics.json_lowering import NONFINITE, STRING, possessive_string
from udfx.semantics.json_structure import bounded_container_pattern

if TYPE_CHECKING:
    from udfx.semantics.json_lowering import JsonPrimitives


def bounded_integer_container() -> str:
    digits = sys.get_int_max_str_digits()
    integer = r"-?[0-9]" + (f"{{1,{digits}}}+(?![0-9])" if digits else "++")
    token = rf"(?:{possessive_string(STRING)}|true|false|null|{integer}|[, :\t\r\n]++)"
    return bounded_container_pattern(token, min(32, sys.getrecursionlimit() // 2 - 1))


def validate(
    primitives: JsonPrimitives,
    value: Any,
    native: Any,
    *,
    complete: Any = None,
    bounded: Any = None,
) -> Any:
    text = primitives.text
    structure = primitives.replace(value, STRING, '""')
    trailing = r",[ \t\r\n]*[\]}]"
    exact = text.when(
        text.matches(value, trailing), ~text.matches(structure, trailing), text.literal(True)
    )
    normalized = primitives.replace(structure, r"(?:\A|[\[,: \t\r\n])(?:NaN|-?Infinity)", "")
    exact = exact & text.when(
        text.matches(value, NONFINITE), ~text.matches(normalized, NONFINITE), text.literal(True)
    )
    exact = exact & within_limits(primitives, value, structure)
    if complete is not None:
        exact = complete & exact
    if bounded is not None:
        exact = bounded | exact
    return text.coalesce(native & exact, text.literal(False))
