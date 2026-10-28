from __future__ import annotations

import polars as pl

from udfx.backends.polars.rules import RULES, when
from udfx.backends.polars.slicing import SEQUENCE
from udfx.backends.polars.strings import list_contains
from udfx.backends.registration import Lowered, register_lists
from udfx.frontend.core.comprehensions import lowercase_words
from udfx.semantics.lists import ListPrimitives, SetPrimitives
from udfx.semantics.sequences import take_prefix
from udfx.types import Kind

for kind in (Kind.INT, Kind.BOOL):
    RULES("__contains__", Kind.LIST, kind)(list_contains)


@RULES("list", Kind.LIST)
def copy(value: pl.Expr) -> pl.Expr:
    return value


@RULES("__add__", Kind.LIST, Kind.LIST)
def concatenate(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.list.concat(right)


@RULES("any", Kind.LIST)
def any_value(value: pl.Expr) -> pl.Expr:
    return value.list.any()


@RULES("all", Kind.LIST)
def all_values(value: pl.Expr) -> pl.Expr:
    return value.list.all()


@RULES("sum", Kind.LIST)
def total(value: pl.Expr) -> Lowered[pl.Expr]:
    result = value.cast(pl.List(pl.Int128)).list.sum().cast(pl.Int64, strict=False)
    return Lowered(result, overflow=result.is_null() & value.is_not_null())


@RULES("sum_float", Kind.LIST)
def float_total(value: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(pl.lit(0.0), pl.lit(False))


@RULES("sorted", Kind.LIST)
def ordered(value: pl.Expr) -> pl.Expr:
    return value.list.sort()


@RULES("frequency_order", Kind.LIST, Kind.INT)
def frequency_order(value: pl.Expr, count: pl.Expr) -> pl.Expr:
    element = pl.element()
    runs = element.sort().rle()
    keys = runs.struct.field("value")
    indexes = pl.int_range(pl.len()).sort_by(element, maintain_order=True)
    first = indexes.gather(element.sort().search_sorted(keys))
    ranked = value.list.eval(
        keys.sort_by(runs.struct.field("len"), first, descending=[True, False])
    )
    return take_prefix(SEQUENCE, ranked, count)


RULES.rewrite("frequency_order", Kind.LIST, Kind.INT)(lowercase_words)


@RULES("__contains__", Kind.LIST, Kind.FLOAT)
def float_contains(value: pl.Expr, item: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(list_contains(value, item), item.is_not_nan())


PRIMITIVES = ListPrimitives(
    literal=pl.lit,
    when=when,
    length=lambda value: value.list.len().cast(pl.Int64),
    repeat=lambda value, count: value.repeat_by(count).list.eval(pl.element().explode()),
    range=pl.int_ranges,
)


@RULES("list", Kind.STR)
def characters(value: pl.Expr) -> pl.Expr:
    return value.str.split("")


register_lists(
    RULES,
    PRIMITIVES,
    SetPrimitives(
        literal=pl.lit,
        length=lambda value: value.list.len().cast(pl.Int64),
        distinct=lambda value: value.list.unique(),
        sort=lambda value: value.list.sort(),
        contains=lambda value, item: value.list.contains(item),
        intersection=lambda left, right: left.list.set_intersection(right),
        union=lambda left, right: left.list.set_union(right),
        concatenate=concatenate,
        difference=lambda left, right: left.list.set_difference(right),
    ),
)
