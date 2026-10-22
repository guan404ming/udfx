from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Any

from udfx.semantics.numbers import RoundPrimitives

EXPONENTIAL_LIMIT = math.log(sys.float_info.max)


def transcendental(
    primitives: RoundPrimitives, is_nan: Callable[[Any], Any], name: str, value: Any
) -> tuple[Any, Any]:
    zero = primitives.literal(0.0)
    one = primitives.literal(1.0)
    infinity = primitives.literal(math.inf)
    missing = is_nan(value)
    if name in {"log", "log2", "log10", "log1p"}:
        boundary = primitives.literal(-1.0) if name == "log1p" else zero
        neutral = zero if name == "log1p" else one
        exact = (value <= boundary) | (value == neutral) | (value == infinity) | missing
        result = primitives.when(
            value == boundary,
            primitives.literal(-math.inf),
            primitives.when(
                (value < boundary) | missing,
                primitives.literal(math.nan),
                primitives.when(value == neutral, value if name == "log1p" else zero, value),
            ),
        )
        return result, exact
    exact = (value == zero) | (primitives.abs(value) == infinity) | missing
    result = primitives.when(
        value == zero,
        value if name == "expm1" else one,
        primitives.when(
            value == primitives.literal(-math.inf),
            primitives.literal(-1.0) if name == "expm1" else zero,
            value,
        ),
    )
    return result, exact
