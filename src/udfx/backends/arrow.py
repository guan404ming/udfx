from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pyarrow as pa
from pyarrow import compute

from udfx.backends.base import Backend, Gather
from udfx.program import Program
from udfx.types import Kind, Type

_SCALARS: dict[Kind, pa.DataType] = {
    Kind.DATETIME: pa.timestamp("us"),
    Kind.JSON: pa.string(),
    Kind.STR: pa.string(),
    Kind.INT: pa.int64(),
    Kind.FLOAT: pa.float64(),
    Kind.BOOL: pa.bool_(),
    Kind.NONE: pa.null(),
}


def dtype(type: Type, *, string_type: pa.DataType | None = None) -> pa.DataType:
    if type.kind is Kind.LIST:
        if type.item is None:
            raise TypeError("list type needs an item type")
        return pa.list_(dtype(type.item, string_type=string_type))
    if string_type is not None and type.kind in {Kind.STR, Kind.JSON}:
        return string_type
    return _SCALARS[type.kind]


def make_gather(columns: Sequence[pa.Array]) -> Gather:
    def gather(pending: Sequence[int]) -> Sequence[Sequence[object]]:
        everything = bool(columns) and len(pending) == len(columns[0])
        selected = columns if everything else [column.take(pending) for column in columns]
        return [
            (
                column.cast(pa.timestamp("us", tz=column.type.tz))
                if pa.types.is_timestamp(column.type)
                else column
            ).to_pylist()
            for column in selected
        ]

    return gather


def fallback(
    backend: Backend[Any],
    program: Program,
    flags: pa.Array,
    native: pa.Array,
    columns: Sequence[pa.Array],
) -> pa.Array:
    count = cast(pa.BooleanArray, flags).true_count
    if count == len(native):
        return native
    missing = (
        None
        if count == 0
        else compute.invert(cast(pa.BooleanArray, compute.fill_null(flags, pa.scalar(False))))
    )
    pending = (
        range(len(native))
        if missing is None
        else cast("list[int]", compute.indices_nonzero(missing).to_pylist())
    )
    values = backend.evaluate_pending_rows(program, pending, make_gather(columns))
    replacements = pa.array(values, type=native.type)
    if missing is None:
        return replacements
    if pa.types.is_list(native.type):
        indices = list(range(len(native)))
        for offset, index in enumerate(pending, len(native)):
            indices[index] = offset
        return pa.concat_arrays([native, replacements]).take(indices)
    return compute.replace_with_mask(native, missing, replacements)
