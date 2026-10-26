import json
import re
from collections.abc import Callable
from typing import Any

from udfx.semantics.literals import TextPrimitives

SPACE = r"[ \t\r\n]*"
STRING = r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt])*"'
ESCAPES = {"\\" + key: json.loads(f'"\\{key}"') for key in '"\\/bfnrt'}


def patterns(possessive: bool) -> tuple[str, str, str]:
    suffix = "+" if possessive else ""
    space = SPACE + suffix
    string = STRING.replace('*"', f'*{suffix}"')
    array = (
        rf"{space}\[(?:{space}{string}(?:{space},{space}{string})*{suffix})?{suffix}"
        rf"{space}\]{space}"
    )
    return string, array, f"{space}{string}{space}"


def tokens(value: str) -> list[str]:
    return re.findall(STRING, value)


def attribute_types(name: str) -> tuple[str, ...]:
    return tuple(
        kind.__name__
        for kind in (str, list, dict, int, float, bool, type(None))
        if hasattr(kind, name)
    )


def valid(value: str) -> bool:
    try:
        json.loads(value)
    except (ValueError, RecursionError):
        return False
    return True


def has_type(value: str, name: str) -> bool:
    try:
        return type(json.loads(value)).__name__ == name
    except (ValueError, RecursionError):
        return False


def native_shape(primitives: TextPrimitives, value: Any, possessive: bool) -> Any:
    _, array, string = patterns(possessive)
    return primitives.matches(value, rf"\A(?:{array}|{string})\z")


def lower_type(
    primitives: TextPrimitives, value: Any, name: Any, possessive: bool
) -> tuple[Any, Any]:
    if not isinstance(name, str):
        return primitives.literal(False), primitives.literal(False)
    _, array, string = patterns(possessive)
    pattern = {"str": string, "list": array}.get(name)
    result = (
        primitives.literal(False)
        if pattern is None
        else primitives.matches(value, rf"\A{pattern}\z")
    )
    return result, native_shape(primitives, value, possessive)


def encode_string(value: str | None, ensure_ascii: bool) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii)


def lower_string(
    primitives: TextPrimitives,
    concatenate: Callable[[Any, Any], Any],
    value: Any,
    ensure_ascii: Any,
) -> tuple[Any, Any]:
    if not isinstance(ensure_ascii, bool):
        return primitives.literal(""), primitives.literal(False)
    characters = ["\\", '"', *map(chr, range(32))]
    if ensure_ascii:
        characters.append(chr(127))
    escaped = value
    for character in characters:
        escaped = primitives.replace(escaped, character, json.dumps(character)[1:-1])
    quoted = concatenate(concatenate(primitives.literal('"'), escaped), primitives.literal('"'))
    present = primitives.not_null(value)
    exact = ~present | ~primitives.matches(value, r"[^\x00-\x7f]") if ensure_ascii else None
    return primitives.when(present, quoted, primitives.literal("null")), exact
