from pathlib import Path

from udfx.types import FLOAT, INT, STR
from workloads.models import Join, Workload
from workloads.native import YEAR_MONTH


def delay_band(delay: float | None, cancelled: float | None) -> str:
    if cancelled == 1.0:
        return "cancelled"
    if delay is None:
        return "unknown"
    if delay < 15.0:
        return "on time"
    if delay < 60.0:
        return "delayed"
    return "severe"


def clock_minutes(clock: int | None) -> int | None:
    if clock is None:
        return None
    return (clock // 100) * 60 + clock % 100


def route(origin: str | None, destination: str | None) -> str | None:
    if origin is None or destination is None:
        return None
    return origin + " → " + destination


def year_month(date: str | None) -> str | None:
    if date is None:
        return None
    return date[:7]


def load(root: Path) -> list[Workload]:
    table = root / "flights" / "flights.parquet"
    join = Join(root / "flights" / "airports.parquet", "origin", "airport", "city")
    return [
        Workload(
            "flights",
            "delay_band",
            delay_band,
            {"delay": FLOAT.or_none(), "cancelled": FLOAT.or_none()},
            STR,
            table,
            {"delay": "arrival_delay", "cancelled": "cancelled"},
            join=join,
        ),
        Workload(
            "flights",
            "clock_minutes",
            clock_minutes,
            {"clock": INT.or_none()},
            INT.or_none(),
            table,
            {"clock": "departure_time"},
            "carrier",
        ),
        Workload(
            "flights",
            "route",
            route,
            {"origin": STR.or_none(), "destination": STR.or_none()},
            STR.or_none(),
            table,
            {"origin": "origin", "destination": "destination"},
            "carrier",
        ),
        Workload(
            "flights",
            "year_month",
            year_month,
            {"date": STR.or_none()},
            STR.or_none(),
            table,
            {"date": "date"},
            "carrier",
            native=YEAR_MONTH,
        ),
    ]
