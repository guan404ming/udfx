from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow
import pytest

from udfx.types import DATETIME, FLOAT, INT, STR
from workloads import catalog, storage
from workloads.models import Workload

SUITE = "sample"
ROWS = 32
CODES = ["11385", " 10001 ", "00000", "1234", "\uff11" * 5, "abcde", None, "60601"]
PLACES = ["LGA", "ATL", "ORD", None]
CLOCKS = [851, 2359, 0, None, 1230]
STARTED = datetime(2024, 1, 1, 0, 0, 0, 123456)


def postal_code(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if len(value) != 5 or not value.isascii() or not value.isdigit() or value == "00000":
        return None
    return value


def clock_minutes(clock: int | None) -> int | None:
    if clock is None:
        return None
    return (clock // 100) * 60 + clock % 100


def route(origin: str | None, destination: str | None) -> str | None:
    if origin is None or destination is None:
        return None
    return origin + " → " + destination


def resolution_hours(created: datetime | None, closed: datetime | None) -> float | None:
    if created is None or closed is None:
        return None
    return (closed - created).total_seconds() / 3600.0


def table() -> pyarrow.Table:
    closed = [STARTED + timedelta(minutes=97 * index) for index in range(ROWS)]
    return pyarrow.table(
        {
            "code": [CODES[index % len(CODES)] for index in range(ROWS)],
            "clock": [CLOCKS[index % len(CLOCKS)] for index in range(ROWS)],
            "origin": [PLACES[index % len(PLACES)] for index in range(ROWS)],
            "destination": [PLACES[index // 8] for index in range(ROWS)],
            "created": [None if index % 7 == 3 else STARTED for index in range(ROWS)],
            "closed": pyarrow.array(closed, pyarrow.timestamp("us")),
            "group": ["even" if index % 2 else "odd" for index in range(ROWS)],
        }
    )


def load(
    root: Path, cases: set[str] | None = None, *, load_functions: bool = True
) -> list[Workload]:
    path = root / SUITE / "rows.parquet"
    workloads = [
        Workload(
            SUITE,
            "route",
            route,
            {"origin": STR.or_none(), "destination": STR.or_none()},
            STR.or_none(),
            path,
            {"origin": "origin", "destination": "destination"},
            "group",
        ),
        Workload(
            SUITE,
            "postal_code",
            postal_code,
            {"value": STR.or_none()},
            STR.or_none(),
            path,
            {"value": "code"},
            "group",
        ),
        Workload(
            SUITE,
            "clock_minutes",
            clock_minutes,
            {"clock": INT.or_none()},
            INT.or_none(),
            path,
            {"clock": "clock"},
            "group",
        ),
        Workload(
            SUITE,
            "resolution_hours",
            resolution_hours,
            {"created": DATETIME.or_none(), "closed": DATETIME.or_none()},
            FLOAT.or_none(),
            path,
            {"created": "created", "closed": "closed"},
        ),
    ]
    return [
        workload if load_functions else replace(workload, function=None)
        for workload in workloads
        if cases is None or workload.identifier in cases
    ]


def register(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(catalog.SUITES, SUITE, load)
    monkeypatch.setattr(storage, "read_manifest", lambda suite: {"name": suite, "files": []})
