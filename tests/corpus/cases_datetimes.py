from datetime import datetime, timedelta

from udfx.types import DATETIME, STR

from . import Case

CASES = [
    Case(
        "datetime_age_days",
        lambda left, right: (left - right).days,
        {"left": DATETIME, "right": DATETIME},
        "native",
    ),
    Case(
        "datetime_elapsed_seconds",
        lambda left, right: (left - right).seconds,
        {"left": DATETIME, "right": DATETIME},
        "native",
    ),
    Case(
        "datetime_total_seconds",
        lambda left, right: (left - right).total_seconds(),
        {"left": DATETIME, "right": DATETIME},
        "native",
    ),
    Case(
        "datetime_shift_forward",
        lambda value: value + timedelta(days=1, hours=2, minutes=3, seconds=4),
        {"value": DATETIME},
        "native",
    ),
    Case(
        "datetime_shift_backward",
        lambda value: value - timedelta(hours=12),
        {"value": DATETIME},
        "native",
    ),
    Case("datetime_year", lambda value: value.year, {"value": DATETIME}, "native"),
    Case("datetime_month", lambda value: value.month, {"value": DATETIME}, "native"),
    Case("datetime_weekday", lambda value: value.weekday(), {"value": DATETIME}, "native"),
    Case(
        "datetime_format", lambda value: value.strftime("%Y-%m-%d"), {"value": DATETIME}, "native"
    ),
    Case(
        "datetime_parse_slice",
        lambda text: datetime.strptime(text[0:10], "%Y-%m-%d"),
        {"text": STR},
        "native",
    ),
    Case(
        "datetime_row",
        lambda row: row.time.hour if row.time is not None else -1,
        {"time": DATETIME.or_none()},
        "native",
    ),
]
