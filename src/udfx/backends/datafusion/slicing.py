from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace as replace_record

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.arrow import dtype
from udfx.backends.datafusion.rules import RULES, length, when
from udfx.backends.datafusion.strings import characters, join, list_len
from udfx.backends.registration import register_padding
from udfx.semantics.facts import result_type
from udfx.semantics.padding import PaddingPrimitives
from udfx.semantics.sequences import (
    SequencePrimitives,
    SlicePrimitives,
    slice_sequence,
)
from udfx.types import INT64_MAX, Kind, Type

SLICE = SlicePrimitives(
    maximum=INT64_MAX,
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
    head=lambda value, size: functions.array_slice(value, 1, size),
    tail=lambda value, size: functions.array_slice(value, -size, 2**31 - 1),
    drop_head=lambda value, size: functions.array_slice(value, min(1 + size, 2**31 - 1), 2**31 - 1),
    drop_tail=lambda value, size: functions.array_slice(value, 1, -size - 1),
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
    drop_head=lambda value, size: functions.right(value, -size),
    drop_tail=lambda value, size: functions.left(value, -size),
    reverse=functions.reverse,
    stride=lambda value, step: join(
        lit(""), SEQUENCE.stride(characters(value), min(step, 2**31 - 1))
    ),
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
    text: Expr,
    start: Expr | int | None,
    stop: Expr | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Expr:
    return slice_sequence(TEXT, text, start, stop, step, result=result)


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
    container: Expr,
    start: Expr | int | None,
    stop: Expr | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Expr:
    return slice_sequence(SEQUENCE, container, start, stop, step, result=result)


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
