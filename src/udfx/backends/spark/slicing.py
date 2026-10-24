from __future__ import annotations

from dataclasses import replace

from pyspark.sql import Column, functions

from udfx.backends.registration import register_padding
from udfx.backends.spark.rules import INT32_MAX, RULES, when
from udfx.backends.spark.strings import characters, join, length, list_len
from udfx.semantics.facts import result_type, split_separator
from udfx.semantics.padding import PaddingPrimitives
from udfx.semantics.sequences import (
    SequencePrimitives,
    SlicePrimitives,
    slice_sequence,
    stride_pattern,
    token_stride_pattern,
)
from udfx.types import Kind, Type


def bounded_native_index(index: Column) -> Column:
    bounded = functions.least(
        functions.greatest(index, functions.lit(-INT32_MAX)), functions.lit(INT32_MAX - 1)
    )
    return functions.when(bounded >= 0, bounded + 1).otherwise(bounded).cast("int")


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(text: Column, index: Column) -> Column:
    return functions.substring(text, bounded_native_index(index), functions.lit(1))


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: Column, index: Column) -> Column:
    return functions.try_element_at(container, bounded_native_index(index))


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


def stride_text(value: Column, step: int) -> Column:
    pattern = stride_pattern(step)
    if pattern is not None:
        return functions.regexp_replace(value, pattern, "$1")
    return join(functions.lit(""), SEQUENCE.stride(characters(value), step))


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
    stride=stride_text,
    bind=SEQUENCE.bind,
    extract=functions.regexp_extract,
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
    facts={"result": result_type, "separator": split_separator},
)
def list_slice(
    container: Column,
    start: Column | int | None,
    stop: Column | int | None,
    step: int | None = None,
    *,
    result: Type,
    separator: str | None = None,
) -> Column:
    sequence = SEQUENCE
    dynamic = isinstance(start, Column) or isinstance(stop, Column)
    if separator is not None and step is not None and abs(step) > 1 and dynamic:
        pattern = token_stride_pattern(separator, abs(step))
        if pattern is not None:
            sequence = replace(
                sequence,
                stride=lambda values, count: functions.when(
                    functions.size(values) == 0, functions.array().cast("array<string>")
                ).otherwise(
                    functions.regexp_extract_all(
                        functions.concat(
                            functions.lit(separator * count), join(functions.lit(separator), values)
                        ),
                        functions.lit(pattern),
                        functions.lit(1),
                    )
                ),
            )
    return slice_sequence(sequence, container, start, stop, step, result=result)


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
