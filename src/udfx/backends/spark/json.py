from collections.abc import Callable

from pyspark.sql import Column, functions

from udfx.backends.registration_json import register_json_values
from udfx.backends.spark.backend import SparkBackend
from udfx.backends.spark.bindings import bind
from udfx.backends.spark.conversions import TEXT
from udfx.backends.spark.regex_categories import category
from udfx.backends.spark.rules import RULES, when
from udfx.backends.spark.runtime import call
from udfx.semantics.json_encoding import (
    JsonEncodingPrimitives,
    encode_spaced_words,
    encode_strings,
)
from udfx.semantics.json_lowering import (
    STRING,
    JsonPrimitives,
    possessive_string,
)
from udfx.semantics.json_paths import initial_object_pattern
from udfx.semantics.json_structure import complete_integer_document
from udfx.semantics.json_validation import bounded_integer_container, validate
from udfx.semantics.text import whitespace
from udfx.types import Type


def replace(value: Column, pattern: str, replacement: str) -> Column:
    if pattern == STRING:
        pattern = possessive_string(pattern)
    return functions.regexp_replace(value, pattern, replacement)


def valid(value: Column, accepted: Column | None = None) -> Column:
    return validate(
        NATIVE,
        value,
        functions.try_parse_json(value).isNotNull() if accepted is None else accepted,
        complete=complete_integer_document(
            TEXT, replace, value, functions.get_json_object(value, "$")
        ),
        bounded=value.rlike(bounded_integer_container()),
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


def field_parts(selected: Column, type: Type) -> tuple[Column, Column, Column]:
    present = selected.isNotNull()
    missing = functions.is_variant_null(selected) | ~present
    output = functions.try_variant_get(selected, "$", "string")
    output = when(missing, functions.lit(None).cast("string"), output)
    typed = functions.schema_of_variant(selected) == functions.lit("STRING")
    typed = typed | (missing if type.nullable else ~present)
    return output, present, typed


def typed_field(
    value: Column, key: Column, type: Type, indexes: tuple[int, ...]
) -> tuple[Column, Column, Column]:
    def projected(output: Column, present: Column, exact: Column) -> Column:
        return functions.struct(
            output.alias("value"), present.alias("present"), exact.alias("exact")
        )

    packed = project_field(value, key, type, indexes, projected)
    return packed["value"], packed["present"], packed["exact"]


def project_field(
    value: Column,
    key: Column,
    type: Type,
    indexes: tuple[int, ...],
    finish: Callable[[Column, Column, Column], Column],
) -> Column:
    if any(position >= 2**31 for position in indexes):
        return finish(
            functions.lit(None).cast("string"), functions.lit(False), functions.lit(False)
        )
    path = "$" + "".join(f"[{position}]" for position in indexes)
    mapping = functions.try_variant_get(
        functions.try_parse_json(value), path, "map<string,variant>"
    )
    inputs = functions.struct(mapping.alias("mapping"), key.alias("key"), value.alias("source"))

    def projected(inputs: Column) -> Column:
        output, present, exact = field_parts(
            functions.try_element_at(inputs["mapping"], inputs["key"]), type
        )
        return finish(
            output, present, exact & valid(inputs["source"], inputs["mapping"].isNotNull())
        )

    return bind(inputs, projected)


def index(value: Column, position: Column) -> Column:
    path = functions.concat(functions.lit("$[*]["), position.cast("string"), functions.lit("]"))
    return extract(value, path)


def object_at_path(value: Column, indexes: tuple[int, ...]) -> tuple[Column, Column | None]:
    if (pattern := initial_object_pattern(indexes)) is not None:
        return functions.coalesce(value.rlike(pattern), functions.lit(False)), None
    if any(position >= 2**31 for position in indexes):
        return functions.lit(False), None
    path = "$" + "".join(f"[{position}]" for position in indexes)
    parsed = functions.try_parse_json(value)
    return functions.try_variant_get(parsed, path, "struct<>").isNotNull(), parsed.isNotNull()


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
    project_field=project_field,
)


def encode(value: Column, ensure_ascii: bool, *, pretty: bool = True) -> Column:
    return functions.to_json(
        value,
        {
            "pretty": str(pretty).lower(),
            "writeNonAsciiCharacterAsCodePoint": str(ensure_ascii).lower(),
        },
    )


ENCODING = JsonEncodingPrimitives(
    text=TEXT,
    encode=encode,
    replace=functions.regexp_replace,
    tokens=lambda value, pattern: functions.regexp_extract_all(value, functions.lit(pattern), 0),
    transform=functions.transform,
    lowercase=functions.lower,
    join=lambda value: functions.array_join(value, ""),
    contains=Column.contains,
    bind=bind,
)


def encode_words(value: Column, ensure_ascii: bool) -> Column:
    if (
        runtime := call(
            "json_words", value, functions.lit(ensure_ascii), functions.lit(whitespace())
        )
    ) is not None:
        return runtime
    normalized = functions.trim(
        functions.regexp_replace(value, category("CATEGORY_SPACE") + "+", " ")
    )
    encoded = encode(functions.array(normalized), ensure_ascii, pretty=False)
    return encode_spaced_words(ENCODING, encoded, ensure_ascii)


def encode_values(value: Column, ensure_ascii: bool, uppercase_free: bool = False) -> Column:
    if (
        not (ensure_ascii and uppercase_free)
        and (runtime := call("json_strings", value, functions.lit(ensure_ascii))) is not None
    ):
        return runtime
    return encode_strings(ENCODING, value, ensure_ascii, uppercase_free)


register_json_values(
    RULES,
    SparkBackend().literal,
    TEXT,
    NATIVE,
    encode_values,
    encode_words=encode_words,
)
