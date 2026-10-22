from __future__ import annotations

import math
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any

from udfx.types import Kind

PYTHON_TYPES = {
    Kind.DATETIME: datetime,
    Kind.STR: str,
    Kind.INT: int,
    Kind.FLOAT: float,
    Kind.BOOL: bool,
    Kind.NONE: type(None),
    Kind.LIST: list,
}


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def not_missing(value: object) -> bool:
    return not is_missing(value)


def coalesce(value: object, default: object) -> object:
    return default if value is None else value


def missing_expression(
    is_null: Callable[[Any], Any], is_nan: Callable[[Any], Any], value: Any
) -> Any:
    return is_null(value) | is_nan(value)


def numpy_scalar(name: str, *arguments: object) -> object:
    numpy = sys.modules["numpy"]
    with numpy.errstate(all="ignore"):
        value = getattr(numpy, name)(*arguments)
    return python_scalar(value)


def python_scalar(value: object) -> object:
    kind = type(value)
    return (
        bool(value) if kind.__module__ == "numpy" and kind.__name__ in ("bool", "bool_") else value
    )
