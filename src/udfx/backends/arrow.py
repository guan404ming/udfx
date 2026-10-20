from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pyarrow as pa
from pyarrow import compute

from udfx.backends.base import Gather
from udfx.types import Kind, Type

_SCALARS: dict[Kind, pa.DataType] = {
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


def gather(columns: Sequence[pa.Array]) -> Gather:
    return lambda pending: [column.take(pending).to_pylist() for column in columns]


def pending(flags: pa.Array) -> list[int]:
    exact = cast(pa.BooleanArray, compute.fill_null(flags, pa.scalar(False)))
    missing = compute.invert(exact)
    return cast("list[int]", compute.indices_nonzero(missing).to_pylist())
