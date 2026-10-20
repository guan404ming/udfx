from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

FLOAT_INT_MAX = 2**53

Binary = Callable[[Any, Any], Any]

ARITHMETIC: dict[str, Binary] = {
    "__add__": operator.add,
    "__sub__": operator.sub,
    "__mul__": operator.mul,
}

COMPARISONS: dict[str, Binary] = {
    "__lt__": operator.lt,
    "__le__": operator.le,
    "__gt__": operator.gt,
    "__ge__": operator.ge,
}
