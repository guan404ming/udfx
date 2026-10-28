from __future__ import annotations

import duckdb
from duckdb import sqltypes

from udfx.types import Kind, Type

_SCALARS: dict[Kind, sqltypes.DuckDBPyType] = {
    Kind.DATETIME: sqltypes.TIMESTAMP,
    Kind.JSON: sqltypes.VARCHAR,
    Kind.STR: sqltypes.VARCHAR,
    Kind.INT: sqltypes.BIGINT,
    Kind.FLOAT: sqltypes.DOUBLE,
    Kind.BOOL: sqltypes.BOOLEAN,
    Kind.NONE: sqltypes.SQLNULL,
}


def dtype(type: Type) -> sqltypes.DuckDBPyType:
    if type.kind is Kind.LIST:
        if type.item is None:
            raise TypeError("list type needs an item type")
        return duckdb.list_type(dtype(type.item))
    return _SCALARS[type.kind]
