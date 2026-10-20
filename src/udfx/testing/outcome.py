from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from udfx.types import Type, fits


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
        value = function(*arguments)
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
    if isinstance(first, list) and isinstance(second, list):
        left = cast("list[object]", first)
        right = cast("list[object]", second)
        return len(left) == len(right) and all(map(same, left, right))
    return type(first) is type(second) and first == second
