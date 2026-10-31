from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from udfx.semantics.literals import TextPrimitives
from udfx.types import BOOL, FLOAT, INT, STR, Type

UNICODE = (
    r"u(?:[0-9a-cA-Ce-fE-F][0-9a-fA-F]{3}|[dD][0-7][0-9a-fA-F]{2}|"
    r"[dD][89aAbB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2})"
)
STRING = rf'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|{UNICODE}))*"'
NONFINITE = r"(?i:\b(?:nan|inf(?:inity)?)\b)"
TYPES = {
    "dict": r"\A[ \t\r\n]*\{",
    "list": r"\A[ \t\r\n]*\[",
    "str": r'\A[ \t\r\n]*"',
    "int": r"\A[ \t\r\n]*-?[0-9]+[ \t\r\n]*\z",
    "float": (
        r"\A[ \t\r\n]*(?:-?(?:[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?|"
        r"[0-9]+[eE][+-]?[0-9]+|Infinity)|NaN)[ \t\r\n]*\z"
    ),
    "bool": r"\A[ \t\r\n]*(?:true|false)[ \t\r\n]*\z",
    "NoneType": r"\A[ \t\r\n]*null[ \t\r\n]*\z",
}


def possessive_string(pattern: str) -> str:
    character = r'[^"\\\x00-\x1f]'
    return pattern.replace(character, f"(?>{character}+)").replace('*"', '*+"')


@dataclass(frozen=True)
class JsonPrimitives:
    text: TextPrimitives
    byte_length: Callable[[Any], Any]
    replace: Callable[[Any, str, str], Any]
    valid: Callable[[Any], Any]
    sanitize: Callable[[Any], Any]
    field: Callable[[Any, Any], Any]
    index: Callable[[Any, Any], Any]
    items: Callable[[Any], Any]
    size: Callable[[Any], Any]
    unique: Callable[[Any], Any]
    decode: Callable[[Any], Any]
    strings: Callable[[Any, Any], Any]
    longest_digits: Callable[[Any], Any]
    possessive: bool = False
    field_or_null: Callable[[Any, Any], Any] | None = None
    typed_field: Callable[[Any, Any, Type, tuple[int, ...]], tuple[Any, Any, Any]] | None = None
    object_at_path: Callable[[Any, tuple[int, ...]], tuple[Any, Any | None]] | None = None
    project_field: (
        Callable[[Any, Any, Type, tuple[int, ...], Callable[[Any, Any, Any], Any]], Any] | None
    ) = None


def accepted(primitives: JsonPrimitives, value: Any, validated: bool = False) -> Any:
    return primitives.text.literal(True) if validated else primitives.valid(value)


def safe(primitives: JsonPrimitives, value: Any, validated: bool = False) -> Any:
    return value if validated else primitives.sanitize(value)


def has_type(primitives: JsonPrimitives, value: Any, name: str) -> Any:
    text = primitives.text
    if (pattern := TYPES.get(name)) is None:
        return text.literal(False)
    return text.coalesce(text.matches(value, pattern), text.literal(False))


def get(
    primitives: JsonPrimitives,
    value: Any,
    key: Any,
    default: Any,
    *,
    validated: bool = False,
    default_validated: bool = False,
    default_none: bool = False,
) -> tuple[Any, Any]:
    text = primitives.text
    document = safe(primitives, value, validated)
    field = primitives.field_or_null if default_none else None
    output, exact = (field or primitives.field)(document, key)
    default_exact = accepted(primitives, default, default_validated)
    default = safe(primitives, default, default_validated)
    output = text.when(text.not_null(key), output, default)
    return (
        text.coalesce(output, default),
        accepted(primitives, value, validated)
        & primitives.unique(document)
        & exact
        & default_exact,
    )


def result(
    primitives: JsonPrimitives, value: Any, type: Type, validated: bool = False
) -> tuple[Any, Any]:
    text = primitives.text
    document = safe(primitives, value, validated)
    exact = has_type(primitives, value, type.kind.value)
    if type == STR or type == STR.or_none():
        output = primitives.decode(document)
    elif type.kind is INT.kind:
        output = text.try_int(document)
    elif type.kind is FLOAT.kind:
        output = text.try_float(document)
    elif type.kind is BOOL.kind:
        output = text.matches(document, r"\A[ \t\r\n]*true[ \t\r\n]*\z")
    else:
        return text.literal(None), text.literal(False)
    exact = exact & text.not_null(output)
    missing = has_type(primitives, value, "NoneType")
    if type.nullable:
        exact = exact | missing
        output = text.when(missing, text.literal(None), output)
    return output, accepted(primitives, value, validated) & exact


def item(
    primitives: JsonPrimitives,
    value: Any,
    key: Any,
    *,
    integer: bool,
    validated: bool,
    container: str | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    text = primitives.text
    document = safe(primitives, value, validated)
    array = has_type(primitives, value, "list")
    mapping = has_type(primitives, value, "dict")
    exact = accepted(primitives, value, validated)
    if integer:
        size = primitives.size(document)
        position = text.when(key < text.literal(0), size + key, key)
        outside = text.coalesce(
            (position < text.literal(0)) | (position >= size), text.literal(False)
        )
        position = text.when(outside, text.literal(0), position)
        output = primitives.index(document, position)
        exact = (
            None
            if validated and container in {"list", "dict"}
            else exact & ~has_type(primitives, value, "str")
        )
        invalid = (array & ~text.not_null(key)) | ~(array | mapping)
        return output, exact, invalid, mapping, array & outside
    output, guard = primitives.field(document, key)
    exact = exact & primitives.unique(document) & guard
    return (
        output,
        exact,
        ~mapping,
        mapping & (~text.not_null(key) | ~text.not_null(output)),
        text.literal(False),
    )


def iterable(primitives: JsonPrimitives, value: Any) -> Any:
    return (
        has_type(primitives, value, "list")
        | has_type(primitives, value, "dict")
        | has_type(primitives, value, "str")
    )
