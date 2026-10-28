import duckdb
from duckdb import Expression

from udfx.backends.duckdb.backend import DuckDBBackend
from udfx.backends.duckdb.conversions import TEXT
from udfx.backends.duckdb.rules import RULES, column, constant, function, length, when
from udfx.backends.registration_json import register_json_values
from udfx.frontend.core.json import serialized_ascii_lists
from udfx.semantics.json_lowering import JsonPrimitives
from udfx.types import Type


def replace(value: Expression, pattern: str, replacement: str) -> Expression:
    return function(
        "regexp_replace", value, constant(pattern), constant(replacement), constant("g")
    )


def pointer(key: Expression) -> Expression:
    return function(
        "||",
        constant("/"),
        function(
            "replace",
            function("replace", key, constant("~"), constant("~0")),
            constant("/"),
            constant("~1"),
        ),
    )


def field(value: Expression, key: Expression) -> tuple[Expression, Expression]:
    return function("json_extract", value, pointer(key)).cast("VARCHAR"), constant(True)


def typed_field(
    value: Expression, key: Expression, type: Type, indexes: tuple[int, ...]
) -> tuple[Expression, Expression, Expression]:
    path = pointer(key)
    if indexes:
        path = function("||", constant("".join(f"/{position}" for position in indexes)), path)
    kind = function("json_type", value, path)
    output = function("json_extract_string", value, path)
    present = kind.isnotnull()
    exact = kind == constant("VARCHAR")
    if type.nullable:
        exact = exact | (kind == constant("NULL"))
    return output, present, ~present | exact


def index(value: Expression, position: Expression) -> Expression:
    return function("json_extract", value, position.cast("BIGINT")).cast("VARCHAR")


def object_at_path(value: Expression, indexes: tuple[int, ...]) -> Expression:
    path = constant("".join(f"/{position}" for position in indexes))
    return duckdb.CoalesceOperator(
        function("json_type", value, path) == constant("OBJECT"), constant(False)
    )


def unique(value: Expression) -> Expression:
    keys = function("json_keys", value)
    return length(keys) == length(function("list_distinct", keys))


def longest_digits(value: Expression) -> Expression:
    digits = function("regexp_extract_all", value, constant("[0-9]+"))
    lengths = function(
        "list_transform",
        digits,
        duckdb.LambdaExpression("udfx digits", length(column("udfx digits"))),
    )
    return duckdb.CoalesceOperator(function("list_max", lengths), constant(0))


NATIVE = JsonPrimitives(
    text=TEXT,
    byte_length=lambda value: function("strlen", value),
    replace=replace,
    valid=lambda value: function("json_valid", value),
    sanitize=lambda value: when(function("json_valid", value), value, constant("null")),
    field=field,
    index=index,
    items=lambda value: function("json_extract", value, constant("$[*]")).cast("VARCHAR[]"),
    size=lambda value: function("json_array_length", value).cast("BIGINT"),
    unique=unique,
    decode=lambda value: function("json_extract_string", value, constant("$")),
    strings=lambda value, exact: function(
        "from_json", when(exact, value, constant("[]")), constant('["VARCHAR"]')
    ),
    longest_digits=longest_digits,
    typed_field=typed_field,
    object_at_path=object_at_path,
)

register_json_values(
    RULES, DuckDBBackend().literal, TEXT, NATIVE, serialize_strings=serialized_ascii_lists
)
