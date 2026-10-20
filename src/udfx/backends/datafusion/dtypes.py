# pyright: reportUnknownParameterType=false
from __future__ import annotations

import pyarrow as pa

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
