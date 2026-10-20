from __future__ import annotations

import polars as pl

from udfx.types import Kind, Type

_SCALARS: dict[Kind, pl.DataType] = {
    Kind.STR: pl.String(),
    Kind.INT: pl.Int64(),
    Kind.FLOAT: pl.Float64(),
    Kind.BOOL: pl.Boolean(),
    Kind.NONE: pl.Null(),
}


def dtype(type: Type) -> pl.DataType:
    if type.kind is Kind.LIST:
        if type.item is None:
            raise TypeError("list type needs an item type")
        return pl.List(dtype(type.item))
    return _SCALARS[type.kind]
