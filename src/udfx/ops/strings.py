from __future__ import annotations

from udfx.ops.registry import Op, register
from udfx.types import STR

register(Op(".lower", (STR,), STR, str.lower))
register(Op(".upper", (STR,), STR, str.upper))
