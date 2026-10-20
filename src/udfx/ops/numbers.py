from __future__ import annotations

import operator

from udfx.ops.registry import Op, register
from udfx.types import BOOL, INT

register(Op("__add__", (INT, INT), INT, operator.add))
register(Op("__gt__", (INT, INT), BOOL, operator.gt))
