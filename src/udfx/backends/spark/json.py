from functools import partial

from pyspark.sql import Column, functions

from udfx.backends.registration_json import register_json_values
from udfx.backends.spark.backend import SparkBackend
from udfx.backends.spark.bindings import bind
from udfx.backends.spark.conversions import TEXT
from udfx.backends.spark.rules import RULES, when
from udfx.semantics.json_encoding import JsonEncodingPrimitives, encode_strings
from udfx.semantics.json_lowering import (
    STRING,
    JsonPrimitives,
    complete_integer_document,
    possessive_string,
)
from udfx.types import Type


def replace(value: Column, pattern: str, replacement: str) -> Column:
    if pattern == STRING:
        pattern = possessive_string(pattern)
    return functions.regexp_replace(value, pattern, replacement)


def valid(value: Column) -> Column:
    parsed = functions.try_parse_json(value)
    return parsed.isNotNull() & complete_integer_document(
        TEXT, replace, value, functions.get_json_object(value, "$")
    )


def extract(value: Column, path: Column) -> Column:
    wrapped = functions.concat(functions.lit("["), value, functions.lit("]"))
    return functions.call_function("get_json_object", wrapped, path)


def field_path(key: Column, root: str) -> tuple[Column, Column]:
    exact = key.isNull() | ((key != "") & ~key.rlike(r"['\\*\x00-\x1f]"))
    key = when(exact, key, functions.lit(""))
    path = functions.concat(functions.lit(root + "['"), key, functions.lit("']"))
    return path, exact


def field_or_null(value: Column, key: Column) -> tuple[Column, Column]:
    path, exact = field_path(key, "$[*]")
    return extract(value, path), exact


def field(value: Column, key: Column) -> tuple[Column, Column]:
    output, exact = field_or_null(value, key)
    present = functions.array_contains(functions.json_object_keys(value), key)
    return functions.coalesce(output, functions.when(present, functions.lit("null"))), exact


def typed_field(
    value: Column, key: Column, type: Type, indexes: tuple[int, ...]
) -> tuple[Column, Column, Column]:
    if any(position >= 2**31 for position in indexes):
        return functions.lit(None).cast("string"), functions.lit(False), functions.lit(False)
    path, exact = field_path(key, "$" + "".join(f"[{position}]" for position in indexes))
    selected = functions.try_variant_get(functions.try_parse_json(value), path, "variant")

    def projected(selected: Column) -> Column:
        present = selected.isNotNull()
        missing = functions.is_variant_null(selected) | ~present
        output = functions.try_variant_get(selected, "$", "string")
        output = when(missing, functions.lit(None).cast("string"), output)
        typed = functions.schema_of_variant(selected) == functions.lit("STRING")
        typed = typed | (missing if type.nullable else ~present)
        return functions.struct(
            output.alias("value"), present.alias("present"), typed.alias("exact")
        )

    packed = bind(selected, projected)
    return packed["value"], packed["present"], exact & packed["exact"]


def index(value: Column, position: Column) -> Column:
    path = functions.concat(functions.lit("$[*]["), position.cast("string"), functions.lit("]"))
    return extract(value, path)


def object_at_path(value: Column, indexes: tuple[int, ...]) -> Column:
    if any(position >= 2**31 for position in indexes):
        return functions.lit(False)
    path = "$" + "".join(f"[{position}]" for position in indexes)
    return functions.try_variant_get(functions.try_parse_json(value), path, "struct<>").isNotNull()


def items(value: Column) -> Column:
    values = functions.from_json(value, "array<string>")
    return functions.transform(values, lambda item, position: index(value, position))


NATIVE = JsonPrimitives(
    text=TEXT,
    byte_length=functions.octet_length,
    replace=replace,
    valid=valid,
    sanitize=lambda value: value,
    field=field,
    index=index,
    items=items,
    size=functions.json_array_length,
    unique=lambda value: functions.lit(True),
    decode=lambda value: functions.get_json_object(value, "$"),
    strings=lambda value, exact: functions.from_json(value, "array<string>"),
    longest_digits=lambda value: functions.aggregate(
        functions.regexp_extract_all(value, functions.lit("[0-9]+"), 0),
        functions.lit(0),
        lambda longest, item: functions.greatest(longest, functions.length(item)),
    ),
    possessive=True,
    field_or_null=field_or_null,
    typed_field=typed_field,
    object_at_path=object_at_path,
)

ENCODING = JsonEncodingPrimitives(
    text=TEXT,
    encode=lambda value, ensure_ascii: functions.to_json(
        value,
        {"pretty": "true", "writeNonAsciiCharacterAsCodePoint": str(ensure_ascii).lower()},
    ),
    replace=functions.regexp_replace,
    tokens=lambda value, pattern: functions.regexp_extract_all(value, functions.lit(pattern), 0),
    transform=functions.transform,
    lowercase=functions.lower,
    join=lambda value: functions.array_join(value, ""),
    contains=Column.contains,
    bind=bind,
)

register_json_values(RULES, SparkBackend().literal, TEXT, NATIVE, partial(encode_strings, ENCODING))
