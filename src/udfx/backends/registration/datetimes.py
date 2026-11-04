from __future__ import annotations

from collections.abc import Callable
from typing import Any

from udfx.backends.rules import Lowered, RuleSet
from udfx.semantics.numbers.operations import COMPARISONS, Binary
from udfx.semantics.temporal.dates import ATTRIBUTES as DATE_ATTRIBUTES
from udfx.semantics.temporal.dates import (
    DateArithmeticPrimitives,
    date_shift_exceeds,
    difference_dates,
    shift_date,
)
from udfx.semantics.temporal.datetimes import (
    ATTRIBUTES,
    DatetimeArithmeticPrimitives,
    DatetimePrimitives,
    datetime_field,
    format_datetime,
    parse_datetime,
    shift_datetime,
    shift_overflows,
)
from udfx.types import Kind


def register_datetimes[E](
    rules: RuleSet[E], primitives: DatetimePrimitives, equal: Binary, to_date: Callable[[E], E]
) -> None:
    for kind, attributes in ((Kind.DATE, DATE_ATTRIBUTES), (Kind.DATETIME, ATTRIBUTES)):
        for name in (*attributes, "weekday", "isoweekday", "toordinal"):
            rules(f".{name}", kind)(
                lambda value, name=name: datetime_field(primitives, value, name)
            )
        for name, compare in COMPARISONS.items():
            rules(name, kind, kind)(compare)
        rules("__eq__", kind, kind)(equal)
        rules("__ne__", kind, kind)(lambda left, right: ~equal(left, right))
        rules("bool", kind)(lambda value: ~primitives.is_null(value))
    rules(".date", Kind.DATETIME)(to_date)
    rules(".strftime", Kind.DATETIME, Kind.STR, literals=(1,))(
        lambda value, format: Lowered(*format_datetime(primitives, value, format))
    )
    rules("datetime.strptime", Kind.STR, Kind.STR, literals=(1,))(
        lambda text, format: Lowered(*parse_datetime(primitives, text, format))
    )

    def parses(text: E, format: Any) -> Lowered[E]:
        parsed, exact = parse_datetime(primitives, text, format)
        return Lowered(~primitives.is_null(parsed), exact)

    rules("datetime_parses", Kind.STR, Kind.STR, literals=(1,))(parses)


def register_datetime_arithmetic[E](
    rules: RuleSet[E], primitives: DatetimeArithmeticPrimitives
) -> None:
    rules("datetime_difference", Kind.DATETIME, Kind.DATETIME)(primitives.difference)
    rules("datetime_shift", Kind.DATETIME, Kind.INT)(
        lambda value, offset: shift_datetime(primitives, value, offset)
    )
    rules("datetime_shift_overflows", Kind.DATETIME, Kind.INT)(
        lambda value, offset: shift_overflows(primitives, primitives.to_microseconds(value), offset)
    )


def register_date_arithmetic[E](rules: RuleSet[E], primitives: DateArithmeticPrimitives) -> None:
    rules("date_difference", Kind.DATE, Kind.DATE)(
        lambda left, right: difference_dates(primitives, left, right)
    )
    rules("date_shift", Kind.DATE, Kind.INT)(
        lambda value, days: shift_date(primitives, value, days)
    )
    rules("date_shift_overflows", Kind.DATE, Kind.INT)(
        lambda value, days: date_shift_exceeds(primitives, primitives.to_ordinal(value), days)
    )
