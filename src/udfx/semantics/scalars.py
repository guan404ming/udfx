from __future__ import annotations

import math
import sys
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from udfx.types import Kind

PYTHON_TYPES = {
    Kind.DATE: date,
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


def numpy_values(name: str, *arguments: Any) -> Any:
    numpy = sys.modules["numpy"]
    with numpy.errstate(all="ignore"):
        return getattr(numpy, name)(*arguments)


def python_power(base: Any, exponent: Any) -> float | None:
    try:
        found = base**exponent
    except (OverflowError, ZeroDivisionError, TypeError):
        return None
    return found if isinstance(found, float) else math.nan


def numpy_scalar(name: str, *arguments: object) -> object:
    return python_scalar(numpy_values(name, *arguments))


def python_scalar(value: object) -> object:
    kind = type(value)
    if kind.__module__ != "numpy":
        return value
    if kind.__name__ in ("bool", "bool_"):
        return bool(value)
    return int(value) if kind.__name__.startswith(("int", "uint")) else value  # pyright: ignore[reportArgumentType]
