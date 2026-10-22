from __future__ import annotations

from typing import Any

from udfx.backends.rules import Lowered, RuleSet
from udfx.semantics.datetimes import (
    ATTRIBUTES,
    DatetimeArithmeticPrimitives,
    DatetimePrimitives,
    datetime_field,
    format_datetime,
    parse_datetime,
    shift_datetime,
    shift_overflows,
)
from udfx.semantics.numbers import COMPARISONS, Binary
from udfx.types import Kind


def register_datetimes[E](rules: RuleSet[E], primitives: DatetimePrimitives, equal: Binary) -> None:
    for name in (*ATTRIBUTES, "weekday", "isoweekday", "toordinal"):
        rules(f".{name}", Kind.DATETIME)(
            lambda value, name=name: datetime_field(primitives, value, name)
        )
    for name, compare in COMPARISONS.items():
        rules(name, Kind.DATETIME, Kind.DATETIME)(compare)
    rules("__eq__", Kind.DATETIME, Kind.DATETIME)(equal)
    rules("__ne__", Kind.DATETIME, Kind.DATETIME)(lambda left, right: ~equal(left, right))
    rules("bool", Kind.DATETIME)(lambda value: ~primitives.is_null(value))
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
    rules("datetime_difference", Kind.DATETIME, Kind.DATETIME)(
        lambda left, right: primitives.to_microseconds(left) - primitives.to_microseconds(right)
    )
    rules("datetime_shift", Kind.DATETIME, Kind.INT)(
        lambda value, offset: shift_datetime(primitives, value, offset)
    )
    rules("datetime_shift_overflows", Kind.DATETIME, Kind.INT)(
        lambda value, offset: shift_overflows(primitives, primitives.to_microseconds(value), offset)
    )
