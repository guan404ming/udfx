from collections.abc import Callable
from typing import Any

from udfx.semantics.json_lowering import STRING, TYPES
from udfx.semantics.literals import TextPrimitives


def bounded_container_pattern(token: str, depth: int) -> str:
    content = token + "*+"
    for _ in range(depth - 1):
        content = rf"(?:{token}|[\[{{]{content}[\]}}])*+"
    return rf"\A[ \t\r\n]*+[\[{{]{content}[\]}}][ \t\r\n]*+\z"


def complete_integer_document(
    text: TextPrimitives, replace: Callable[[Any, str, str], Any], value: Any, serialized: Any
) -> Any:
    structure = replace(value, STRING, '""')
    original = replace(structure, r"[ \t\r\n]", "")
    output = replace(replace(serialized, STRING, '""'), r"[ \t\r\n]", "")
    complete = ~text.matches(original, r"[0-9][.eE]|\\u[dD][89a-fA-F]") & (
        text.char_length(original) == text.char_length(output)
    )
    complete = text.when(
        text.matches(value, TYPES["str"]),
        text.matches(structure, r'\A[ \t\r\n]*""[ \t\r\n]*\z'),
        complete,
    )
    token = r'(?:""|true|false|null|-?[0-9]++|[, :\t\r\n]++)'
    return text.when(
        text.matches(structure, bounded_container_pattern(token, 32)), text.literal(True), complete
    )
