from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from udfx.semantics.literals import TextPrimitives


@dataclass(frozen=True)
class JsonEncodingPrimitives:
    text: TextPrimitives
    encode: Callable[[Any, bool], Any]
    replace: Callable[[Any, str, str], Any]
    tokens: Callable[[Any, str], Any]
    transform: Callable[[Any, Callable[[Any], Any]], Any]
    lowercase: Callable[[Any], Any]
    join: Callable[[Any], Any]
    contains: Callable[[Any, str], Any]
    bind: Callable[[Any, Callable[[Any], Any]], Any]


def encode_strings(primitives: JsonEncodingPrimitives, value: Any, ensure_ascii: bool) -> Any:
    text = primitives.text
    encoded = primitives.replace(primitives.encode(value, ensure_ascii), r"\A\[ ", "[")
    encoded = primitives.replace(encoded, r" \]\z", "]")
    if ensure_ascii:
        encoded = text.replace(encoded, chr(127), r"\u007f")
    return primitives.bind(encoded, lambda value: normalize_escapes(primitives, value))


def normalize_escapes(primitives: JsonEncodingPrimitives, encoded: Any) -> Any:
    text = primitives.text
    parts = primitives.tokens(encoded, r"(?:\\u[0-9a-fA-F]{4})++|(?:[^\\]++|\\[^u])++")
    normalized = primitives.transform(
        parts,
        lambda part: text.when(text.matches(part, r"\A\\u"), primitives.lowercase(part), part),
    )
    result = text.when(primitives.contains(encoded, r"\u"), primitives.join(normalized), encoded)
    return text.coalesce(result, text.literal("null"))
