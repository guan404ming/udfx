from __future__ import annotations

import operator
from datetime import datetime

from udfx.ir import Node
from udfx.ops.registry import Fault, Op, apply, register
from udfx.semantics.datetimes import (
    ATTRIBUTES,
    datetime_difference,
    datetime_shift,
    datetime_shift_overflows,
    strptime_parses,
)
from udfx.semantics.numbers import COMPARISONS
from udfx.types import BOOL, DATETIME, INT, STR


def parse_faults(text: Node, format: Node) -> tuple[Fault, ...]:
    return (Fault(ValueError, apply("not", apply("datetime_parses", text, format))),)


def shift_faults(value: Node, microseconds: Node) -> tuple[Fault, ...]:
    return (Fault(OverflowError, apply("datetime_shift_overflows", value, microseconds)),)


for name in ATTRIBUTES:
    register(Op(f".{name}", (DATETIME,), INT, operator.attrgetter(name)))

for name in ("weekday", "isoweekday", "toordinal"):
    register(Op(f".{name}", (DATETIME,), INT, getattr(datetime, name)))

for name, compare in COMPARISONS.items():
    register(Op(name, (DATETIME, DATETIME), BOOL, compare))

for name, compare in (("__eq__", operator.eq), ("__ne__", operator.ne)):
    register(Op(name, (DATETIME.or_none(), DATETIME.or_none()), BOOL, compare))

register(Op(".strftime", (DATETIME, STR), STR, datetime.strftime))
register(Op("datetime_parses", (STR, STR), BOOL, strptime_parses))
register(Op("datetime.strptime", (STR, STR), DATETIME, datetime.strptime, parse_faults))


register(Op("datetime_difference", (DATETIME, DATETIME), INT, datetime_difference))
register(Op("datetime_shift", (DATETIME, INT), DATETIME, datetime_shift, shift_faults))
register(Op("datetime_shift_overflows", (DATETIME, INT), BOOL, datetime_shift_overflows))
