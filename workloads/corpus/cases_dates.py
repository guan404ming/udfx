from datetime import date, timedelta

from udfx.types import DATE

from . import Case

CASES = [
    Case(
        "date_calendar_window",
        lambda value: date(2000, 1, 1) <= value < date(2001, 1, 1),
        {"value": DATE},
        "supported",
    ),
    Case(
        "date_signed_day_shift",
        lambda value: value + timedelta(hours=-12) - timedelta(hours=12),
        {"value": DATE},
        "supported",
    ),
    Case(
        "date_elapsed_days",
        lambda left, right: (left - right).days,
        {"left": DATE, "right": DATE},
        "supported",
    ),
    Case(
        "date_nullable_row",
        lambda row: row.value if row.value is not None else date(1970, 1, 1),
        {"value": DATE.or_none()},
        "supported",
    ),
]
