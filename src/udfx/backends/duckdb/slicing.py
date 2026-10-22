from __future__ import annotations

import duckdb
from duckdb import Expression

from udfx.backends.duckdb.rules import (
    RULES,
    bind,
    constant,
    function,
    length,
    when,
)
from udfx.backends.duckdb.strings import join, single_code_point_clusters
from udfx.backends.registration import register_padding
from udfx.semantics.facts import result_type
from udfx.semantics.padding import PaddingPrimitives
from udfx.semantics.sequences import (
    SequencePrimitives,
    SlicePrimitives,
    slice_sequence,
)
from udfx.types import Kind, Type

SLICE = SlicePrimitives(
    maximum=2**32 - 1,
    literal=lambda value: constant(value).cast("BIGINT"),
    when=when,
    coalesce=duckdb.CoalesceOperator,
    clip=lambda value, lower, upper: function("least", function("greatest", value, lower), upper),
)


SEQUENCE = SequencePrimitives(
    bounds=SLICE,
    length=length,
    slice=lambda value, start, size: function(
        "list_slice", value, start + constant(1), start + size
    ),
    head=lambda value, size: function("list_slice", value, constant(1), constant(size)),
    tail=lambda value, size: function(
        "list_slice", value, constant(-size), constant(SLICE.maximum)
    ),
    drop_head=lambda value, size: function(
        "list_slice", value, constant(1 + size), constant(SLICE.maximum)
    ),
    drop_tail=lambda value, size: function("list_slice", value, constant(1), constant(-size - 1)),
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
    drop_head=lambda value, size: function("right", value, constant(-size)),
    drop_tail=lambda value, size: function("left", value, constant(-size)),
    reverse=lambda value: bind(
        value,
        lambda value: when(
            single_code_point_clusters(value),
            function("reverse", value),
            join(constant(""), SEQUENCE.reverse(function("string_split", value, constant("")))),
        ),
    ),
    stride=lambda value, step: join(
        constant(""), SEQUENCE.stride(function("string_split", value, constant("")), step)
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
    text: Expression,
    start: Expression | int | None,
    stop: Expression | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Expression:
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
    container: Expression,
    start: Expression | int | None,
    stop: Expression | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Expression:
    return slice_sequence(SEQUENCE, container, start, stop, step, result=result)


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
