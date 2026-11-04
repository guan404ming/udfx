from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from conftest import frontend_backend_params

from udfx.backends.base import Backend
from udfx.backends.diagnostics import capture_lowering
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.testing.datetime_traps import DATETIME_TRAPS
from udfx.types import DATETIME

EAST = timezone(timedelta(hours=5, minutes=30))
WEST = timezone(timedelta(hours=-8), "Pacific")
CUTOFF = datetime(2000, 1, 1, tzinfo=EAST)
NAMED = ZoneInfo("UTC")
EDGE = datetime.min.replace(tzinfo=EAST)


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: value.replace(tzinfo=UTC).timestamp(),
        lambda value: value.replace(tzinfo=timezone.utc).hour,  # noqa: UP017
        lambda value: value.replace(tzinfo=EAST).timestamp(),
        lambda value: value.replace(tzinfo=EAST).astimezone(UTC).replace(tzinfo=None),
        lambda value: value.replace(tzinfo=WEST).astimezone(tz=EAST).hour,
        lambda value: value.replace(tzinfo=EAST).astimezone(EAST).minute,
        lambda value: value.replace(tzinfo=timezone(timedelta(hours=-3))).astimezone(UTC).day,
        lambda value: value.replace(tzinfo=UTC) < CUTOFF,
        lambda value: value.replace(tzinfo=EAST) >= CUTOFF,
        lambda value: value.replace(tzinfo=UTC) > EDGE,
        lambda value: (value.replace(tzinfo=WEST) - EDGE).days,
        lambda value: value.replace(tzinfo=EAST) == CUTOFF,
        lambda value: value.replace(tzinfo=WEST) != CUTOFF,
        lambda value: value.replace(tzinfo=UTC) == value,
        lambda value: value.year > 2000 and value.replace(tzinfo=UTC) < value,
        lambda value: value.year > 2000 and value - value.replace(tzinfo=UTC),
        lambda value: (value.replace(tzinfo=WEST) - CUTOFF).total_seconds(),
        lambda value: (value.replace(tzinfo=EAST) + timedelta(hours=20)).strftime(
            "%Y-%m-%d %H:%M%z %:z %Z %%z"
        ),
        lambda value: (value.replace(tzinfo=WEST) - timedelta(days=1)).strftime("%Z %H"),
        lambda value: value.replace(tzinfo=EAST).utcoffset().total_seconds(),
        lambda value: value.replace(tzinfo=WEST).tzname(),
        lambda value: value.replace(tzinfo=EAST).dst() is None,
        lambda value: value.replace(tzinfo=EAST).date().toordinal(),
        lambda value: value.replace(tzinfo=EAST).weekday(),
        lambda value: value.replace(tzinfo=EAST).year,
        lambda value: value.replace(tzinfo=EAST).replace(tzinfo=None),
        lambda value: value.replace(tzinfo=EAST).replace(tzinfo=UTC) > CUTOFF,
        lambda value: value.replace(tzinfo=EAST) is None,
        lambda value: not value.replace(tzinfo=EAST),
        lambda value: CUTOFF.hour + CUTOFF.minute,
    ],
)
def test_fixed_offset_datetimes(function: Callable[..., object], backend: Backend[Any]) -> None:
    rows = [(value,) for value in (*DATETIME_TRAPS, None)]
    assert check_function(function, {"value": DATETIME.or_none()}, backend, rows) == []


@pytest.mark.parametrize(
    "function",
    [
        lambda value: value.replace(tzinfo=UTC),
        lambda value: [value.replace(tzinfo=UTC)],
        lambda value: value.replace(tzinfo=NAMED).hour,
        lambda value: value.replace(tzinfo=UTC).astimezone().hour,
        lambda value: value.replace(tzinfo=UTC).astimezone(NAMED).hour,
        lambda value: value.astimezone(UTC).hour,
        lambda value: value.replace(tzinfo=UTC).isoformat(),
        lambda value: value.replace(tzinfo=UTC, hour=1).hour,
        lambda value: str(value.replace(tzinfo=UTC)),
    ],
)
def test_unsupported_zone_forms(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported):
        trace(function, {"value": DATETIME})


@pytest.mark.parametrize("backend", frontend_backend_params())
def test_constant_offsets_stay_native(backend: Backend[Any]) -> None:
    rows = [(value,) for value in DATETIME_TRAPS]
    with capture_lowering() as details:
        assert (
            check_function(
                lambda value: value.replace(tzinfo=UTC) < CUTOFF,
                {"value": DATETIME},
                backend,
                rows,
            )
            == []
        )
    assert details[-1].exact is None
