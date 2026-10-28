import json
import re
from collections.abc import Callable
from typing import Any

from udfx.semantics.json_lowering import TYPES, possessive_string
from udfx.semantics.literals import TextPrimitives
from udfx.semantics.text import whitespace

SPACE = r"[ \t\r\n]*"
STRING = r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt])*"'
ESCAPES = {"\\" + key: json.loads(f'"\\{key}"') for key in '"\\/bfnrt'}
ESCAPE_PATTERNS = (
    (r'(["\\\x08\x09\x0a\x0c\x0d])', r"\\\1"),
    (r"([\x00-\x07\x0b\x0e\x0f])", r"\\u000\1"),
    (r"([\x10-\x1f])", r"\\u001\1"),
)
CONTROL_CHARACTERS = "".join(map(chr, range(32)))
CONTROL_REPLACEMENTS = "".join(json.dumps(point)[-2] for point in CONTROL_CHARACTERS)


def non_ascii_word_pattern() -> str:
    allowed = "".join(re.escape(character) for character in whitespace() if ord(character) > 127)
    return r"[^\x00-\x7f" + allowed + "]"


def patterns(possessive: bool) -> tuple[str, str, str]:
    suffix = "+" if possessive else ""
    space = SPACE + suffix
    string = possessive_string(STRING) if possessive else STRING
    array = (
        rf"{space}\[(?:{space}{string}(?:{space},{space}{string})*{suffix})?{suffix}"
        rf"{space}\]{space}"
    )
    return string, array, f"{space}{string}{space}"


def tokens(value: str) -> list[str]:
    return re.findall(STRING, value)


def simple_string(value: str) -> bool:
    return re.fullmatch(SPACE + STRING + SPACE, value) is not None


def simple_array(value: str) -> bool:
    _, array, _ = patterns(False)
    return re.fullmatch(array, value) is not None


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


def lower_valid(primitives: TextPrimitives, value: Any, exact: Any) -> tuple[Any, Any]:
    prefix = rf'\A{SPACE}(?:[\[{{"]|-?(?:[0-9]|Infinity)|true|false|null|NaN)'
    possible = primitives.coalesce(primitives.matches(value, prefix), primitives.literal(False))
    return possible, ~possible | exact


def lower_type(
    primitives: TextPrimitives,
    value: Any,
    name: Any,
    possessive: bool,
    validated: bool = False,
) -> tuple[Any, Any]:
    if not isinstance(name, str):
        return primitives.literal(False), primitives.literal(False)
    pattern = TYPES.get(name)
    result = (
        primitives.literal(False)
        if pattern is None
        else primitives.coalesce(primitives.matches(value, pattern), primitives.literal(False))
    )
    return result, None if validated else native_shape(primitives, value, possessive)


def encode(value: object, ensure_ascii: bool) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii)


def lower_string(
    primitives: TextPrimitives,
    concatenate: Callable[[Any, Any], Any],
    value: Any,
    ensure_ascii: Any,
    encode: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any]:
    if not isinstance(ensure_ascii, bool):
        return primitives.literal(""), primitives.literal(False)
    if encode is None:
        characters = ["\\", '"', *map(chr, range(32))]
        escaped = value
        for character in characters:
            escaped = primitives.replace(escaped, character, json.dumps(character)[1:-1])
        quoted = concatenate(concatenate(primitives.literal('"'), escaped), primitives.literal('"'))
        plain = concatenate(concatenate(primitives.literal('"'), value), primitives.literal('"'))
        quoted = primitives.when(primitives.matches(value, r'["\\\x00-\x1f]'), quoted, plain)
    else:
        quoted = encode(value)
    if ensure_ascii:
        quoted = primitives.replace(quoted, chr(127), r"\u007f")
    present = primitives.not_null(value)
    exact = ~present | ~primitives.matches(value, r"[^\x00-\x7f]") if ensure_ascii else None
    return primitives.when(present, quoted, primitives.literal("null")), exact


def lower_decoded_string(
    primitives: TextPrimitives,
    value: Any,
    decode: Callable[[Any], Any],
    nullable: bool,
    validated: bool = False,
) -> tuple[Any, Any]:
    if validated:
        return decode(value), None
    _, _, string = patterns(False)
    pattern = rf"\A(?:{string}|{SPACE}null{SPACE})\z" if nullable else rf"\A{string}\z"
    exact = primitives.matches(value, pattern)
    safe = primitives.when(exact, value, primitives.literal('""'))
    return decode(safe), exact


def lower_decoded_strings(
    primitives: TextPrimitives,
    value: Any,
    decode: Callable[[Any, Any], Any],
    possessive: bool = False,
) -> tuple[Any, Any]:
    _, array, _ = patterns(possessive)
    exact = primitives.matches(value, rf"\A{array}\z")
    return decode(value, exact), exact
