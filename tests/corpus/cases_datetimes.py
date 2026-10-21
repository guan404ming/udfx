from datetime import datetime

from udfx.types import DATETIME, STR

from . import Case

CASES = [
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
