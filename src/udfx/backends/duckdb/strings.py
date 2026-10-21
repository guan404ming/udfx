from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache, partial

import duckdb
import pyarrow as pa
from duckdb import Expression

from udfx.backends.duckdb.rules import (
    RULES,
    column,
    constant,
    function,
    length,
    null_safe_equal,
    when,
)
from udfx.backends.registration import (
    Lowered,
    RuleResult,
    register_padding,
    register_string_predicates,
)
from udfx.semantics import text
from udfx.semantics.text import (
    CASE_METHODS,
    CLUSTER_JOINERS,
    SIGMA,
    CaseDrift,
    CasePrimitives,
    PaddingPrimitives,
    SearchPrimitives,
    SequencePrimitives,
    SlicePrimitives,
    SplitPrimitives,
    case_expression,
    case_guard,
    count_occurrences,
    non_whitespace_pattern,
    reverse_find,
    slice_sequence,
    split_limited,
    split_right,
    strip_pattern,
    whitespace,
    word_prefix_pattern,
)
from udfx.types import Kind, Type

POINT = "udfx point"
ITEM = "udfx item"
INDEX = "udfx index"


def native_recase(method: str, samples: Sequence[str]) -> list[str]:
    table = pa.table({POINT: pa.array(list(samples))})
    rows = duckdb.from_arrow(table).select(case_expression(CASE, column(POINT), method)).fetchall()
    return [row[0] for row in rows]


@cache
def case_drift(method: str) -> CaseDrift:
    engine = f"duckdb {duckdb.__version__}"
    return text.case_drift(method, partial(native_recase, method), engine)


def recase(value: Expression, method: str) -> RuleResult[Expression]:
    expr = case_expression(CASE, value, method)
    sigma = function("contains", value, constant(SIGMA))
    exact = case_guard(
        case_drift(method),
        lambda pattern: function("regexp_matches", value, constant(pattern)),
        sigma,
    )
    return expr if exact is None else Lowered(expr, exact)


def trim(text: Expression, characters: str, method: str) -> Expression:
    return function(
        "regexp_replace",
        text,
        constant(strip_pattern(characters, method)),
        constant(""),
        constant("g"),
    )


@RULES(".strip", Kind.STR)
@RULES(".strip", Kind.STR, Kind.STR)
def strip(text: Expression, characters: Expression | None = None) -> Expression:
    if characters is None:
        return trim(text, whitespace(), "strip")
    return function("trim", text, duckdb.CoalesceOperator(characters, constant(whitespace())))


@RULES(".lstrip", Kind.STR)
@RULES(".lstrip", Kind.STR, Kind.STR)
def lstrip(text: Expression, characters: Expression | None = None) -> Expression:
    if characters is None:
        return trim(text, whitespace(), "lstrip")
    return function("ltrim", text, duckdb.CoalesceOperator(characters, constant(whitespace())))


@RULES(".rstrip", Kind.STR)
@RULES(".rstrip", Kind.STR, Kind.STR)
def rstrip(text: Expression, characters: Expression | None = None) -> Expression:
    if characters is None:
        return trim(text, whitespace(), "rstrip")
    return function("rtrim", text, duckdb.CoalesceOperator(characters, constant(whitespace())))


def split_words(text: Expression) -> Expression:
    return function("regexp_extract_all", text, constant(non_whitespace_pattern()))


RULES(".split", Kind.STR)(split_words)
RULES(".rsplit", Kind.STR)(split_words)


def split(text: Expression, separator: Expression) -> Expression:
    return when(separator.isnull(), split_words(text), function("string_split", text, separator))


RULES(".split", Kind.STR, Kind.STR)(split)


@RULES("last_split", Kind.STR, Kind.STR)
def last_split(text: Expression, separator: Expression) -> Expression:
    return function("split_part", text, separator, constant(-1))


def find(text: Expression, pattern: Expression) -> Expression:
    return function("instr", text, pattern) - constant(1)


RULES(".find", Kind.STR, Kind.STR)(find)


def single_code_point_clusters(text: Expression) -> Expression:
    return ~function("regexp_matches", text, constant(CLUSTER_JOINERS))


SEARCH = SearchPrimitives(constant, when, length, lambda text: function("reverse", text), find)


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: Expression, pattern: Expression) -> Lowered[Expression]:
    found = reverse_find(SEARCH, text, pattern)
    return Lowered(found, single_code_point_clusters(text) & single_code_point_clusters(pattern))


@RULES(".count", Kind.STR, Kind.STR)
def count(text: Expression, pattern: Expression) -> Expression:
    pieces = length(function("string_split", text, pattern)) - constant(1)
    return count_occurrences(SEARCH, text, pattern, pieces)


@RULES(".startswith", Kind.STR, Kind.STR)
def startswith(text: Expression, prefix: Expression) -> Expression:
    return function("starts_with", text, prefix)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(text: Expression, suffix: Expression) -> Expression:
    return function("ends_with", text, suffix)


@RULES(".replace", Kind.STR, Kind.STR, Kind.STR)
def replace(text: Expression, old: Expression, new: Expression) -> Expression:
    escaped = function("replace", new, constant("\\"), constant("\\\\"))
    between = function("regexp_replace", text, constant(""), escaped, constant("g"))
    return when(old == constant(""), between, function("replace", text, old, new))


def join(separator: Expression, words: Expression) -> Expression:
    first = column(INDEX) == constant(1)
    prefixed = when(first, column(ITEM), function("concat", separator, column(ITEM)))
    pieces = function("list_transform", words, duckdb.LambdaExpression((ITEM, INDEX), prefixed))
    return function("array_to_string", pieces, constant(""))


RULES(".join", Kind.STR, Kind.LIST)(join)


@RULES(".join", Kind.STR, Kind.STR)
def join_characters(separator: Expression, text: Expression) -> Expression:
    return join(separator, function("string_split", text, constant("")))


@RULES("__add__", Kind.STR, Kind.STR)
def str_add(left: Expression, right: Expression) -> Expression:
    return function("concat", left, right)


RULES("__eq__", Kind.STR, Kind.STR)(null_safe_equal)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(left: Expression, right: Expression) -> Expression:
    return ~null_safe_equal(left, right)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(text: Expression, item: Expression) -> Expression:
    return function("contains", text, item)


def list_contains(container: Expression, item: Expression) -> Expression:
    return function("list_position", container, item).isnotnull()


RULES("__contains__", Kind.LIST, Kind.STR)(list_contains)


RULES("len", Kind.STR)(length)
RULES("len", Kind.LIST)(length)


def bounded_native_index(container: Expression, index: Expression) -> Expression:
    size = length(container)
    bounded = function("greatest", function("least", index, size), -size - constant(1))
    return when(bounded < constant(0), bounded, bounded + constant(1))


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(text: Expression, index: Expression) -> Expression:
    return function("array_extract", text, bounded_native_index(text, index))


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: Expression, index: Expression) -> Expression:
    return function("list_extract", container, bounded_native_index(container, index))


SLICE = SlicePrimitives(
    maximum=2**32 - 1,
    literal=lambda value: constant(value).cast("BIGINT"),
    when=when,
    coalesce=duckdb.CoalesceOperator,
    clip=lambda value, lower, upper: function("least", function("greatest", value, lower), upper),
)


def bind(
    value: Expression, transform: Callable[[Expression], Expression], name: str = ITEM
) -> Expression:
    return function(
        "list_extract",
        function(
            "list_transform",
            function("list_value", value),
            duckdb.LambdaExpression(name, transform(column(name))),
        ),
        constant(1),
    )


SEQUENCE = SequencePrimitives(
    bounds=SLICE,
    length=length,
    slice=lambda value, start, size: function(
        "list_slice", value, start + constant(1), start + size
    ),
    head=lambda value, size: function(
        "list_slice", value, constant(1), constant(size - 1 if size < 0 else size)
    ),
    tail=lambda value, size: function(
        "list_slice", value, constant(-size if size > 0 else 1 - size), constant(SLICE.maximum)
    ),
    reverse=lambda value: function("list_reverse", value),
    stride=lambda value, step: function(
        "list_slice", value, constant(1), constant(SLICE.maximum), constant(step)
    ),
    bind=lambda value, transform, result: bind(value, transform),
)


TEXT = SequencePrimitives(
    bounds=SLICE,
    length=length,
    slice=lambda value, start, size: function("substring", value, start + constant(1), size),
    head=lambda value, size: function("left", value, constant(size)),
    tail=lambda value, size: function("right", value, constant(size)),
    reverse=lambda value: join(
        constant(""), SEQUENCE.reverse(function("string_split", value, constant("")))
    ),
    stride=lambda value, step: join(
        constant(""), SEQUENCE.stride(function("string_split", value, constant("")), step)
    ),
    bind=SEQUENCE.bind,
)


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
@RULES("slice", Kind.STR, Kind.INT, Kind.INT, Kind.INT)
def str_slice(
    text: Expression,
    start: Expression | int | None,
    stop: Expression | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Expression:
    return slice_sequence(TEXT, text, start, stop, step, result=result)


@RULES("slice", Kind.LIST, Kind.INT, Kind.INT)
@RULES("slice", Kind.LIST, Kind.INT, Kind.INT, Kind.INT)
def list_slice(
    container: Expression,
    start: Expression | int | None,
    stop: Expression | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Expression:
    return slice_sequence(SEQUENCE, container, start, stop, step, result=result)


SPLIT = SplitPrimitives(
    sequence=SEQUENCE,
    bind=lambda values, transform: bind(
        function("struct_pack", *(value.alias(str(index)) for index, value in enumerate(values))),
        lambda value: transform(
            *(
                function("struct_extract", value, constant(str(index)))
                for index in range(len(values))
            )
        ),
        f"udfx split {len(values)}",
    ),
    split=split,
    join=join,
    append=lambda values, value: function("list_append", values, value),
    length=length,
    suffix=lambda value, start: function("substring", value, start + constant(1)),
    word_prefix=lambda value, count: function(
        "array_to_string",
        function(
            "list_slice",
            function("regexp_extract_all", value, constant(word_prefix_pattern())),
            constant(1),
            count,
        ),
        constant(""),
    ),
    lstrip=lstrip,
    is_null=Expression.isnull,
    reverse=lambda value: function("reverse", value),
    restore=lambda values: function(
        "list_transform",
        function("list_reverse", values),
        duckdb.LambdaExpression(ITEM, function("reverse", column(ITEM))),
    ),
)


@RULES(".split", Kind.STR, Kind.STR, Kind.INT)
def split_limit(text: Expression, separator: Expression, maximum: Expression) -> Expression:
    return split_limited(SPLIT, text, separator, maximum)


@RULES(".rsplit", Kind.STR, Kind.STR)
@RULES(".rsplit", Kind.STR, Kind.STR, Kind.INT)
def rsplit(
    text: Expression, separator: Expression, maximum: Expression | None = None
) -> Lowered[Expression]:
    exact = single_code_point_clusters(text) & duckdb.CoalesceOperator(
        single_code_point_clusters(separator), constant(True)
    )
    return Lowered(split_right(SPLIT, text, separator, maximum), exact)


register_string_predicates(
    RULES, lambda value, pattern: function("regexp_matches", value, constant(pattern))
)


for method in CASE_METHODS:
    RULES(f".{method}", Kind.STR)(partial(recase, method=method))


CASE = CasePrimitives(
    lower=lambda value: function("lower", value),
    upper=lambda value: function("upper", value),
    title=lambda value: text.title_words(
        CASE,
        value,
        lambda value, pattern, transform: function(
            "array_to_string",
            function(
                "list_transform",
                function("regexp_extract_all", value, constant(pattern)),
                duckdb.LambdaExpression(ITEM, transform(column(ITEM))),
            ),
            constant(""),
        ),
    ),
    head=lambda value: function("substring", value, constant(1), constant(1)),
    tail=lambda value: function("substring", value, constant(2)),
    concat=lambda *values: function("concat", *values),
    when=when,
    matches=lambda value, pattern: function("regexp_matches", value, constant(pattern)),
    characters=lambda value, transform: function(
        "array_to_string",
        function(
            "list_transform",
            function("string_split", value, constant("")),
            duckdb.LambdaExpression(ITEM, transform(column(ITEM))),
        ),
        constant(""),
    ),
)


register_padding(
    RULES,
    PaddingPrimitives(
        sequence=TEXT,
        literal=constant,
        concat=lambda *values: function("concat", *values),
        repeat=lambda value, count: function("repeat", value, count),
        half=lambda value: function("//", value, constant(2)),
    ),
)
