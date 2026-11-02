from __future__ import annotations

import operator
from datetime import date, datetime

from udfx.ir import Node
from udfx.ops.registry import Fault, Op, apply, register
from udfx.semantics.dates import ATTRIBUTES as DATE_ATTRIBUTES
from udfx.semantics.dates import date_difference, date_shift, date_shift_overflows
from udfx.semantics.datetimes import (
    ATTRIBUTES,
    datetime_difference,
    datetime_shift,
    datetime_shift_overflows,
    strptime_parses,
)
from udfx.semantics.numbers import COMPARISONS
from udfx.types import BOOL, DATE, DATETIME, INT, STR


def parse_faults(text: Node, format: Node) -> tuple[Fault, ...]:
    return (Fault(ValueError, apply("not", apply("datetime_parses", text, format))),)


def shift_faults(value: Node, microseconds: Node) -> tuple[Fault, ...]:
    return (Fault(OverflowError, apply("datetime_shift_overflows", value, microseconds)),)


def date_shift_faults(value: Node, days: Node) -> tuple[Fault, ...]:
    return (Fault(OverflowError, apply("date_shift_overflows", value, days)),)


for type, python_type, attributes in (
    (DATETIME, datetime, ATTRIBUTES),
    (DATE, date, DATE_ATTRIBUTES),
):
    for name in attributes:
        register(Op(f".{name}", (type,), INT, operator.attrgetter(name)))

    for name in ("weekday", "isoweekday", "toordinal"):
        register(Op(f".{name}", (type,), INT, getattr(python_type, name)))

    for name, compare in COMPARISONS.items():
        register(Op(name, (type, type), BOOL, compare))

    for name, compare in (("__eq__", operator.eq), ("__ne__", operator.ne)):
        register(Op(name, (type.or_none(), type.or_none()), BOOL, compare))

register(Op(".date", (DATETIME,), DATE, datetime.date))
register(Op(".strftime", (DATETIME, STR), STR, datetime.strftime))
register(Op("datetime_parses", (STR, STR), BOOL, strptime_parses))
register(Op("datetime.strptime", (STR, STR), DATETIME, datetime.strptime, parse_faults))


register(Op("datetime_difference", (DATETIME, DATETIME), INT, datetime_difference))
register(Op("datetime_shift", (DATETIME, INT), DATETIME, datetime_shift, shift_faults))
register(Op("datetime_shift_overflows", (DATETIME, INT), BOOL, datetime_shift_overflows))
register(Op("date_difference", (DATE, DATE), INT, date_difference))
register(Op("date_shift", (DATE, INT), DATE, date_shift, date_shift_faults))
register(Op("date_shift_overflows", (DATE, INT), BOOL, date_shift_overflows))
