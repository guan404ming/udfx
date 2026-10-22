from __future__ import annotations

from dataclasses import replace as replace_record

import polars as pl

from udfx.backends.polars.rules import RULES, when
from udfx.backends.polars.strings import length, list_len, single_code_point_clusters
from udfx.backends.registration import (
    Lowered,
    register_padding,
)
from udfx.semantics.facts import result_type
from udfx.semantics.padding import PaddingPrimitives
from udfx.semantics.sequences import (
    SequencePrimitives,
    SlicePrimitives,
    reverse_exact,
    slice_sequence,
    stride_pattern,
)
from udfx.types import INT64_MAX, Kind, Type

SLICE = SlicePrimitives(
    maximum=INT64_MAX,
    literal=pl.lit,
    when=when,
    coalesce=pl.Expr.fill_null,
    clip=lambda value, lower, upper: pl.min_horizontal(pl.max_horizontal(value, lower), upper),
)


SEQUENCE = SequencePrimitives(
    bounds=replace_record(SLICE, maximum=2**32 - 1),
    length=list_len,
    slice=lambda value, start, length: value.list.slice(start.fill_null(0), length.fill_null(0)),
    head=lambda value, size: value.list.head(size),
    tail=lambda value, size: value.list.slice(-size),
    drop_head=lambda value, size: value.list.slice(size),
    drop_tail=lambda value, size: value.list.eval(
        pl.element().slice(0, (pl.len().cast(pl.Int64) - size).clip(0))
    ),
    reverse=lambda value: value.list.reverse(),
    stride=lambda value, step: value.list.gather_every(step),
    bind=lambda value, transform, result: (
        pl.concat_list(pl.struct(value.alias("udfx value")))
        .list.eval(transform(pl.element().struct.field("udfx value")))
        .list.first()
    ),
)


def stride_text(value: pl.Expr, step: int) -> pl.Expr:
    pattern = stride_pattern(step)
    if pattern is not None:
        return value.str.replace_all(pattern, "${1}")
    return SEQUENCE.stride(value.str.split(""), min(step, 2**32 - 1)).list.join("")


TEXT = SequencePrimitives(
    bounds=SLICE,
    length=length,
    slice=lambda value, start, size: value.str.slice(start, size),
    head=lambda value, size: value.str.head(size),
    tail=lambda value, size: value.str.tail(size),
    drop_head=lambda value, size: value.str.tail(-size),
    drop_tail=lambda value, size: value.str.head(-size),
    reverse=lambda value: value.str.reverse(),
    stride=stride_text,
    bind=SEQUENCE.bind,
)


@RULES("slice", Kind.STR, Kind.INT, Kind.INT, literals=(1, 2), facts={"result": result_type})
@RULES(
    "slice",
    Kind.STR,
    Kind.INT,
    Kind.INT,
    Kind.INT,
    literals=(1, 2, 3),
    facts={"result": result_type},
)
def str_slice(
    text: pl.Expr,
    start: pl.Expr | int | None,
    stop: pl.Expr | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Lowered[pl.Expr]:
    return Lowered(
        slice_sequence(TEXT, text, start, stop, step, result=result),
        reverse_exact(step, lambda: single_code_point_clusters(text)),
    )


@RULES("slice", Kind.LIST, Kind.INT, Kind.INT, literals=(1, 2), facts={"result": result_type})
@RULES(
    "slice",
    Kind.LIST,
    Kind.INT,
    Kind.INT,
    Kind.INT,
    literals=(1, 2, 3),
    facts={"result": result_type},
)
def list_slice(
    container: pl.Expr,
    start: pl.Expr | int | None,
    stop: pl.Expr | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> pl.Expr:
    return slice_sequence(SEQUENCE, container, start, stop, step, result=result)


register_padding(
    RULES,
    PaddingPrimitives(
        sequence=TEXT,
        literal=pl.lit,
        concat=lambda *values: pl.concat_str(values),
        repeat=lambda value, count: value.repeat_by(count).list.join(""),
        half=lambda value: value // 2,
    ),
)
