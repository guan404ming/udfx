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
        selected = [column.take(pending) for column in columns]
        return [
            (
                column.cast(pa.timestamp("us", tz=column.type.tz))
                if pa.types.is_timestamp(column.type)
                else column
            ).to_pylist()
            for column in selected
        ]

    return gather


def pending(flags: pa.Array) -> list[int]:
    exact = cast(pa.BooleanArray, compute.fill_null(flags, pa.scalar(False)))
    missing = compute.invert(exact)
    return cast("list[int]", compute.indices_nonzero(missing).to_pylist())


def fallback(
    backend: Backend[Any],
    program: Program,
    flags: pa.Array,
    columns: Sequence[pa.Array],
    result: pa.DataType,
) -> pa.Array:
    values = backend.fallback_values(program, len(flags), pending(flags), make_gather(columns))
    return pa.array(values, type=result)
