from __future__ import annotations

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DataType,
    DoubleType,
    LongType,
    NullType,
    StringType,
)

from udfx.types import Kind, Type

_SCALARS: dict[Kind, DataType] = {
    Kind.STR: StringType(),
    Kind.INT: LongType(),
    Kind.FLOAT: DoubleType(),
    Kind.BOOL: BooleanType(),
    Kind.NONE: NullType(),
}


def dtype(type: Type) -> DataType:
    if type.kind is Kind.LIST:
        if type.item is None:
            raise TypeError("list type needs an item type")
        return ArrayType(dtype(type.item))
    return _SCALARS[type.kind]
