from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from udfx.semantics.datetimes import MICROSECONDS_PER_DAY

ATTRIBUTES = ("year", "month", "day")
MINIMUM_ORDINAL = date.min.toordinal()
MAXIMUM_ORDINAL = date.max.toordinal()
MAXIMUM_DIFFERENCE = (MAXIMUM_ORDINAL - MINIMUM_ORDINAL) * MICROSECONDS_PER_DAY


def date_difference(left: date, right: date) -> int:
    return (left - right).days * MICROSECONDS_PER_DAY


def date_shift(value: date, days: int) -> date:
    return value + timedelta(days=days)


def date_shift_overflows(value: date, days: int) -> bool:
    ordinal = value.toordinal()
    return days < MINIMUM_ORDINAL - ordinal or days > MAXIMUM_ORDINAL - ordinal


@dataclass(frozen=True)
class DateArithmeticPrimitives:
    literal: Callable[[Any], Any]
    to_ordinal: Callable[[Any], Any]
    from_ordinal: Callable[[Any], Any]
    select: Callable[[Any, Any, Any], Any]


def difference_dates(primitives: DateArithmeticPrimitives, left: Any, right: Any) -> Any:
    days = primitives.to_ordinal(left) - primitives.to_ordinal(right)
    return days * primitives.literal(MICROSECONDS_PER_DAY)


def date_shift_exceeds(primitives: DateArithmeticPrimitives, ordinal: Any, days: Any) -> Any:
    minimum = primitives.literal(MINIMUM_ORDINAL) - ordinal
    maximum = primitives.literal(MAXIMUM_ORDINAL) - ordinal
    return (days < minimum) | (days > maximum)


def shift_date(primitives: DateArithmeticPrimitives, value: Any, days: Any) -> Any:
    ordinal = primitives.to_ordinal(value)
    safe = primitives.select(
        date_shift_exceeds(primitives, ordinal, days), primitives.literal(0), days
    )
    return primitives.from_ordinal(ordinal + safe)
