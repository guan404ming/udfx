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
    function: Callable[..., object], args: Sequence[object], result: Type | None = None
) -> Outcome:
    try:
        value = function(*args)
    except Exception as error:
        return Raised(type(error))
    if result is not None and not fits(value, result):
        return Raised(OverflowError)
    return Returned(value)


def same(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    if isinstance(a, list) and isinstance(b, list):
        left = cast("list[object]", a)
        right = cast("list[object]", b)
        return len(left) == len(right) and all(map(same, left, right))
    return type(a) is type(b) and a == b
