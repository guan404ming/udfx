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
    Kind.STR: pa.string(),
    Kind.INT: pa.int64(),
    Kind.FLOAT: pa.float64(),
    Kind.BOOL: pa.bool_(),
    Kind.NONE: pa.null(),
}


def dtype(type: Type) -> pa.DataType:
    if type.kind is Kind.LIST:
        if type.item is None:
            raise TypeError("list type needs an item type")
        return pa.list_(dtype(type.item))
    return _SCALARS[type.kind]


def make_gather(columns: Sequence[pa.Array]) -> Gather:
    def gather(pending: list[int]) -> Sequence[Sequence[object]]:
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
    exact = cast(pa.BooleanArray, compute.fill_null(flags, pa.scalar(False)))
    missing = compute.invert(exact)
    pending = cast("list[int]", compute.indices_nonzero(missing).to_pylist())
    if not pending:
        return native
    values = backend.evaluate_pending_rows(program, pending, make_gather(columns))
    replacements = pa.array(values, type=native.type)
    if len(pending) == len(native):
        return replacements
    if pa.types.is_list(native.type):
        indices = list(range(len(native)))
        for offset, index in enumerate(pending, len(native)):
            indices[index] = offset
        return pa.concat_arrays([native, replacements]).take(indices)
    return compute.replace_with_mask(native, missing, replacements)


def python_rows(backend: Backend[Any], program: Program, columns: Sequence[pa.Array]) -> pa.Array:
    pending = list(range(len(columns[0])))
    values = backend.evaluate_pending_rows(program, pending, make_gather(columns))
    return pa.array(values, type=dtype(program.result))
