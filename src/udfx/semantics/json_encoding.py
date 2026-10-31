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


def encode_strings(
    primitives: JsonEncodingPrimitives,
    value: Any,
    ensure_ascii: bool,
    ascii_uppercase_free: bool = False,
) -> Any:
    encoded = primitives.replace(primitives.encode(value, ensure_ascii), r"\A\[ ", "[")
    encoded = primitives.replace(encoded, r" \]\z", "]")
    return primitives.bind(
        escape_delete(primitives.text, encoded, ensure_ascii),
        lambda value: normalize_escapes(primitives, value, ensure_ascii and ascii_uppercase_free),
    )


def encode_spaced_words(
    primitives: JsonEncodingPrimitives, encoded: Any, ensure_ascii: bool
) -> Any:
    text = primitives.text

    def finish(encoded: Any) -> Any:
        separated = text.replace(normalize_escapes(primitives, encoded), " ", '", "')
        return text.when(encoded == text.literal('[""]'), text.literal("[]"), separated)

    return primitives.bind(escape_delete(text, encoded, ensure_ascii), finish)


def escape_delete(text: TextPrimitives, encoded: Any, ensure_ascii: bool) -> Any:
    return text.replace(encoded, chr(127), r"\u007f") if ensure_ascii else encoded


def normalize_escapes(
    primitives: JsonEncodingPrimitives, encoded: Any, ascii_uppercase_free: bool = False
) -> Any:
    text = primitives.text
    if ascii_uppercase_free:
        normalized = primitives.lowercase(encoded)
    else:
        parts = primitives.tokens(encoded, r"(?:\\u[0-9a-fA-F]{4})++|(?:[^\\]++|\\[^u])++")
        normalized = primitives.join(
            primitives.transform(
                parts,
                lambda part: text.when(
                    text.matches(part, r"\A\\u"), primitives.lowercase(part), part
                ),
            )
        )
    result = text.when(primitives.contains(encoded, r"\u"), normalized, encoded)
    return text.coalesce(result, text.literal("null"))
