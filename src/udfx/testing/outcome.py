from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from udfx.semantics.scalars import python_scalar
from udfx.types import Kind, Type, fits


@dataclass(frozen=True)
class Returned:
    value: object


@dataclass(frozen=True)
class Raised:
    kind: type[BaseException]


Outcome = Returned | Raised


def run(
    function: Callable[..., object], arguments: Sequence[object], result: Type | None = None
) -> Outcome:
    try:
        value = python_scalar(function(*arguments))
    except Exception as error:
        return Raised(type(error))
    if result is not None and not fits(value, result):
        return Raised(OverflowError)
    return Returned(value)


def same(first: object, second: object) -> bool:
    if isinstance(first, bool) or isinstance(second, bool):
        return type(first) is type(second) and first == second
    if isinstance(first, float) and isinstance(second, float):
        if math.isnan(first) or math.isnan(second):
            return math.isnan(first) and math.isnan(second)
        return first == second and math.copysign(1.0, first) == math.copysign(1.0, second)
    if isinstance(first, dict) and isinstance(second, dict):
        return list(first) == list(second) and all(same(first[key], second[key]) for key in first)
    if isinstance(first, list) and isinstance(second, list):
        left = cast("list[object]", first)
        right = cast("list[object]", second)
        return len(left) == len(right) and all(map(same, left, right))
    return type(first) is type(second) and first == second


def same_typed(first: object, second: object, type: Type) -> bool:
    if type.kind is Kind.JSON:
        if not isinstance(first, str) or not isinstance(second, str):
            return same(first, second)
        try:
            return same(json.loads(first), json.loads(second))
        except (ValueError, RecursionError):
            return same(first, second)
    if type.item is not None and isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            same_typed(left, right, type.item) for left, right in zip(first, second, strict=True)
        )
    return same(first, second)
