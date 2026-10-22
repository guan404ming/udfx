from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace as replace_record
from functools import cache, partial
from typing import cast

import datafusion
import pyarrow as pa
from datafusion import Expr, col, functions, lit

from udfx.backends.arrow import dtype
from udfx.backends.datafusion.backend import DataFusionBackend
from udfx.backends.datafusion.conversions import TEXT as TRANSLATION
from udfx.backends.datafusion.rules import RULES, length, null_safe_equal, typed_literal, when
from udfx.backends.registration import (
    Lowered,
    RuleResult,
    register_padding,
    register_string_predicates,
    register_translation,
)
from udfx.semantics import text
from udfx.semantics.text import (
    CASE_METHODS,
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
    char_class,
    count_occurrences,
    non_whitespace_pattern,
    reverse_find,
    slice_sequence,
    split_limited,
    split_right,
    whitespace,
)
from udfx.types import STR, Kind, Type, list_of

POINT = "point"


def native_recase(method: str, samples: Sequence[str]) -> list[str]:
    expr = case_expression(CASE, col(POINT), method)
    (recased,) = DataFusionBackend().evaluate(
        [expr], {POINT: STR}, [(sample,) for sample in samples]
    )
    return cast("list[str]", recased)


@cache
def case_drift(method: str) -> CaseDrift:
    engine = f"datafusion {datafusion.__version__}"
    return text.case_drift(method, partial(native_recase, method), engine)


def recase(value: Expr, method: str) -> RuleResult[Expr]:
    expr = case_expression(CASE, value, method)
    sigma = functions.strpos(value, lit(SIGMA)) > lit(0)
    exact = case_guard(
        case_drift(method), lambda pattern: functions.regexp_like(value, lit(pattern)), sigma
    )
    return expr if exact is None else Lowered(expr, exact)


@RULES(".strip", Kind.STR)
@RULES(".strip", Kind.STR, Kind.STR)
def strip(text: Expr, characters: Expr | None = None) -> Expr:
    characters = (
        functions.coalesce(characters, lit(whitespace()))
        if characters is not None
        else lit(whitespace())
    )
    return functions.btrim(text, characters)


@RULES(".lstrip", Kind.STR)
@RULES(".lstrip", Kind.STR, Kind.STR)
def lstrip(text: Expr, characters: Expr | None = None) -> Expr:
    characters = (
        functions.coalesce(characters, lit(whitespace()))
        if characters is not None
        else lit(whitespace())
    )
    return functions.ltrim(text, characters)


@RULES(".rstrip", Kind.STR)
@RULES(".rstrip", Kind.STR, Kind.STR)
def rstrip(text: Expr, characters: Expr | None = None) -> Expr:
    characters = (
        functions.coalesce(characters, lit(whitespace()))
        if characters is not None
        else lit(whitespace())
    )
    return functions.rtrim(text, characters)


def split_words(text: Expr) -> Expr:
    spaced = functions.regexp_replace(text, lit(char_class(whitespace()) + "+"), lit(" "), lit("g"))
    return functions.string_to_array(functions.btrim(spaced, lit(" ")), lit(" "))


RULES(".split", Kind.STR)(split_words)
RULES(".rsplit", Kind.STR)(split_words)


def split(text: Expr, separator: Expr) -> Expr:
    return (
        functions.when(separator.is_null(), split_words(text))
        .when(text == lit(""), functions.make_array(lit("")))
        .otherwise(functions.string_to_array(text, separator))
    )


RULES(".split", Kind.STR, Kind.STR)(split)


@RULES("last_split", Kind.STR, Kind.STR)
def last_split(text: Expr, separator: Expr) -> Expr:
    return functions.split_part(text, separator, lit(-1))


def find(text: Expr, pattern: Expr) -> Expr:
    return functions.strpos(text, pattern).cast(pa.int64()) - lit(1)


RULES(".find", Kind.STR, Kind.STR)(find)


SEARCH = SearchPrimitives(lit, when, length, functions.reverse, find)


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: Expr, pattern: Expr) -> Expr:
    return reverse_find(SEARCH, text, pattern)


@RULES(".count", Kind.STR, Kind.STR)
def count(text: Expr, pattern: Expr) -> Expr:
    removed = length(text) - length(functions.replace(text, pattern, lit("")))
    occurrences = removed / functions.greatest(length(pattern), lit(1))
    return count_occurrences(SEARCH, text, pattern, occurrences)


@RULES(".startswith", Kind.STR, Kind.STR)
def startswith(text: Expr, prefix: Expr) -> Expr:
    return functions.starts_with(text, prefix)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(text: Expr, suffix: Expr) -> Expr:
    return functions.ends_with(text, suffix)


def characters(text: Expr) -> Expr:
    return functions.string_to_array(text, typed_literal(None, STR.or_none()))


@RULES(".replace", Kind.STR, Kind.STR, Kind.STR)
def replace(text: Expr, old: Expr, new: Expr) -> Expr:
    padded = functions.concat(new, functions.array_to_string(characters(text), new), new)
    return (
        functions.when((old == lit("")) & (text == lit("")), new)
        .when(old == lit(""), padded)
        .otherwise(functions.replace(text, old, new))
    )


def join(separator: Expr, words: Expr) -> Expr:
    return functions.array_to_string(words, separator)


RULES(".join", Kind.STR, Kind.LIST)(join)


@RULES(".join", Kind.STR, Kind.STR)
def join_characters(separator: Expr, text: Expr) -> Expr:
    return join(separator, characters(text))


@RULES("__add__", Kind.STR, Kind.STR)
def str_add(left: Expr, right: Expr) -> Expr:
    return functions.concat(left, right)


@RULES("__eq__", Kind.STR, Kind.STR)
def str_eq(left: Expr, right: Expr) -> Expr:
    return null_safe_equal(left, right)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(left: Expr, right: Expr) -> Expr:
    return ~null_safe_equal(left, right)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(text: Expr, item: Expr) -> Expr:
    return functions.strpos(text, item) > lit(0)


def list_contains(container: Expr, item: Expr) -> Expr:
    if container.variant_name() == "Literal":
        values = container.python_value()
        if isinstance(values, pa.ListScalar):
            values = values.as_py()
        if isinstance(values, list):
            found = functions.in_list(item, [lit(value) for value in values if value is not None])
            return functions.coalesce(found, lit(False)) | (item.is_null() & lit(None in values))
    return functions.array_position(container, item).is_not_null()


RULES("__contains__", Kind.LIST, Kind.STR)(list_contains)


@RULES("len", Kind.STR)
def str_len(text: Expr) -> Expr:
    return length(text)


def list_len(container: Expr) -> Expr:
    return functions.cardinality(container).cast(pa.int64())


RULES("len", Kind.LIST)(list_len)


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(text: Expr, index: Expr) -> Expr:
    total = length(text)
    position = functions.when(index < lit(0), index + total).otherwise(index)
    clamped = functions.greatest(functions.least(position, total), lit(-1))
    return functions.substr(text, clamped + lit(1), lit(1))


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: Expr, index: Expr) -> Expr:
    total = list_len(container)
    position = functions.when(index < lit(0), functions.greatest(index, lit(-1) - total)).otherwise(
        functions.least(index, total) + lit(1)
    )
    return functions.array_element(container, position)


SLICE = SlicePrimitives(
    maximum=2**63 - 1,
    literal=lit,
    when=when,
    coalesce=functions.coalesce,
    clip=lambda value, lower, upper: functions.greatest(functions.least(value, upper), lower),
)


def array_transform(values: Expr, transform: Callable[[Expr], Expr], result: pa.DataType) -> Expr:
    nested = functions.array_transform(
        functions.make_array(lit(0)),
        lambda ignored: functions.array_transform(values, transform).cast(pa.list_(result)),
    )
    return functions.array_element(nested.cast(pa.list_(pa.list_(result))), 1)


SEQUENCE = SequencePrimitives(
    bounds=replace_record(SLICE, maximum=2**31 - 1),
    length=list_len,
    slice=lambda value, start, size: functions.array_slice(value, start + lit(1), start + size),
    head=lambda value, size: functions.array_slice(value, 1, size - 1 if size < 0 else size),
    tail=lambda value, size: functions.array_slice(
        value, -size if size > 0 else min(1 - size, 2**31 - 1), 2**31 - 1
    ),
    reverse=functions.array_reverse,
    stride=lambda value, step: functions.array_slice(value, 1, 2**31 - 1, step),
    bind=lambda value, transform, result: functions.array_element(
        array_transform(functions.make_array(value), transform, dtype(result)),
        1,
    ),
)


TEXT = SequencePrimitives(
    bounds=SLICE,
    length=length,
    slice=lambda value, start, size: functions.substr(value, start + lit(1), size),
    head=functions.left,
    tail=functions.right,
    reverse=functions.reverse,
    stride=lambda value, step: join(
        lit(""), SEQUENCE.stride(characters(value), min(step, 2**31 - 1))
    ),
    bind=SEQUENCE.bind,
)


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
@RULES("slice", Kind.STR, Kind.INT, Kind.INT, Kind.INT)
def str_slice(
    text: Expr,
    start: Expr | int | None,
    stop: Expr | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Expr:
    return slice_sequence(TEXT, text, start, stop, step, result=result)


@RULES("slice", Kind.LIST, Kind.INT, Kind.INT)
@RULES("slice", Kind.LIST, Kind.INT, Kind.INT, Kind.INT)
def list_slice(
    container: Expr,
    start: Expr | int | None,
    stop: Expr | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Expr:
    return slice_sequence(SEQUENCE, container, start, stop, step, result=result)


SPLIT = SplitPrimitives(
    sequence=SEQUENCE,
    bind=lambda values, transform: SEQUENCE.bind(
        functions.named_struct([(str(index), value) for index, value in enumerate(values)]),
        lambda value: transform(
            *(functions.get_field(value, str(index)) for index in range(len(values)))
        ),
        list_of(STR),
    ),
    split=split,
    join=join,
    append=functions.array_append,
    length=length,
    suffix=lambda value, start: functions.substr(value, start + lit(1)),
    word_prefix=lambda value, count: functions.substr(
        value,
        lit(1),
        functions.greatest(
            functions.regexp_instr(value, lit(non_whitespace_pattern()), lit(1), count + lit(1))
            - lit(1),
            lit(0),
        ),
    ),
    lstrip=lstrip,
    is_null=Expr.is_null,
    reverse=functions.reverse,
    restore=lambda values: array_transform(
        functions.array_reverse(values), functions.reverse, pa.string()
    ),
)


@RULES(".split", Kind.STR, Kind.STR, Kind.INT)
def split_limit(text: Expr, separator: Expr, maximum: Expr) -> Expr:
    return split_limited(SPLIT, text, separator, maximum)


@RULES(".rsplit", Kind.STR, Kind.STR)
@RULES(".rsplit", Kind.STR, Kind.STR, Kind.INT)
def rsplit(text: Expr, separator: Expr, maximum: Expr | None = None) -> Expr:
    return split_right(SPLIT, text, separator, maximum)


register_string_predicates(RULES, lambda value, pattern: functions.regexp_like(value, lit(pattern)))


for method in CASE_METHODS:
    RULES(f".{method}", Kind.STR)(partial(recase, method=method))


CASE = CasePrimitives(
    lower=functions.lower,
    upper=functions.upper,
    title=functions.initcap,
    head=lambda value: functions.substr(value, lit(1), lit(1)),
    tail=lambda value: functions.substr(value, lit(2)),
    concat=functions.concat,
    when=when,
    matches=lambda value, pattern: functions.regexp_like(value, lit(pattern)),
    characters=lambda value, transform: functions.array_to_string(
        functions.array_transform(characters(value), transform), lit("")
    ),
)


register_padding(
    RULES,
    PaddingPrimitives(
        sequence=TEXT,
        literal=lit,
        concat=functions.concat,
        repeat=functions.repeat,
        half=lambda value: value / lit(2),
    ),
)


register_translation(RULES, TRANSLATION)
