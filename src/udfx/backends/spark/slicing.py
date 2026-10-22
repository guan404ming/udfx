from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.registration import register_padding
from udfx.backends.spark.rules import RULES, when
from udfx.backends.spark.strings import INT32_MAX, characters, join, length, list_len
from udfx.semantics.facts import result_type
from udfx.semantics.padding import PaddingPrimitives
from udfx.semantics.sequences import SequencePrimitives, SlicePrimitives, slice_sequence
from udfx.types import Kind, Type

SLICE = SlicePrimitives(
    maximum=INT32_MAX,
    literal=functions.lit,
    when=when,
    coalesce=functions.coalesce,
    clip=lambda value, lower, upper: functions.least(functions.greatest(value, lower), upper),
)


SEQUENCE = SequencePrimitives(
    bounds=SLICE,
    length=list_len,
    slice=lambda value, start, size: functions.slice(
        value, (start + 1).cast("int"), size.cast("int")
    ),
    head=lambda value, size: functions.slice(value, 1, size),
    tail=lambda value, size: functions.reverse(functions.slice(functions.reverse(value), 1, size)),
    drop_head=lambda value, size: functions.slice(value, 1 + size, INT32_MAX - size),
    drop_tail=lambda value, size: functions.reverse(
        functions.slice(functions.reverse(value), 1 + size, INT32_MAX - size)
    ),
    reverse=functions.reverse,
    stride=lambda value, step: functions.filter(value, lambda item, index: index % step == 0),
    bind=lambda value, transform, result: functions.transform(functions.array(value), transform)[0],
)


TEXT = SequencePrimitives(
    bounds=SLICE,
    length=length,
    slice=lambda value, start, size: functions.substring(
        value, (start + 1).cast("int"), size.cast("int")
    ),
    head=lambda value, size: functions.substring(value, 1, size),
    tail=lambda value, size: functions.substring(value, -size, INT32_MAX),
    drop_head=lambda value, size: functions.substring(value, min(1 + size, INT32_MAX), INT32_MAX),
    drop_tail=lambda value, size: functions.reverse(
        functions.substring(functions.reverse(value), min(1 + size, INT32_MAX), INT32_MAX)
    ),
    reverse=functions.reverse,
    stride=lambda value, step: join(functions.lit(""), SEQUENCE.stride(characters(value), step)),
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
    text: Column,
    start: Column | int | None,
    stop: Column | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Column:
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
    container: Column,
    start: Column | int | None,
    stop: Column | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Column:
    return slice_sequence(SEQUENCE, container, start, stop, step, result=result)


register_padding(
    RULES,
    PaddingPrimitives(
        sequence=TEXT,
        literal=functions.lit,
        concat=functions.concat,
        repeat=lambda value, count: functions.repeat(value, count.cast("int")),
        half=lambda value: functions.call_function("div", value, functions.lit(2)),
    ),
)
