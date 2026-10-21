from __future__ import annotations

import datetime as datetime_module
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import Call
from udfx.testing import check_function
from udfx.testing.strategies import DATETIME_TRAPS
from udfx.types import DATETIME, FLOAT, STR, Type

PARSE = datetime.strptime
BOUND = datetime(1970, 1, 1)


@pytest.mark.parametrize(
    ("function", "name"),
    [
        (lambda value: value.year, ".year"),
        (lambda value: value.month, ".month"),
        (lambda value: value.day, ".day"),
        (lambda value: value.hour, ".hour"),
        (lambda value: value.minute, ".minute"),
        (lambda value: value.second, ".second"),
        (lambda value: value.microsecond, ".microsecond"),
        (lambda value: value.weekday(), ".weekday"),
        (lambda value: value.isoweekday(), ".isoweekday"),
        (lambda value: value.toordinal(), ".toordinal"),
        (lambda value: value.strftime("%Y-%m-%d"), ".strftime"),
    ],
)
def test_datetime_forms(function: Callable[..., object], name: str) -> None:
    program = trace(function, {"value": DATETIME})
    assert isinstance(program.value, Call)
    assert program.value.op.name == name


@pytest.mark.parametrize(
    "function",
    [
        lambda text: datetime.strptime(text, "%Y-%m-%d"),
        lambda text: datetime_module.datetime.strptime(text, "%Y-%m-%d"),
        lambda text: PARSE(text, "%Y-%m-%d"),
    ],
)
def test_strptime_forms(function: Callable[..., object]) -> None:
    program = trace(function, {"text": STR})
    assert isinstance(program.value, Call)
    assert program.value.op.name == "datetime.strptime"
    assert program.result == DATETIME


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: value.year,
        lambda value: value.weekday(),
        lambda value: value.strftime("%Y-%m-%d"),
        lambda value: value.year if value is not None else 0,
        lambda value: value is None,
        lambda value: value == None,  # noqa: E711
        lambda value: bool(value),  # noqa: PLW0108
        lambda value: value == BOUND,
        lambda value: value != BOUND,
        lambda value: value < BOUND,
        lambda value: value <= BOUND,
        lambda value: value > BOUND,
        lambda value: value >= BOUND,
    ],
)
def test_nullable_datetimes(function: Callable[..., object], backend: Backend[Any]) -> None:
    rows = [(value,) for value in (*DATETIME_TRAPS, None)]
    assert check_function(function, {"value": DATETIME.or_none()}, backend, rows) == []


@pytest.mark.parametrize(
    ("function", "schema"),
    [
        (lambda value: datetime.fromtimestamp(value), {"value": FLOAT}),  # noqa: PLW0108
        (lambda value: datetime.utcfromtimestamp(value), {"value": FLOAT}),  # noqa: PLW0108  # pyright: ignore[reportDeprecated]
        (lambda value: value.year(), {"value": DATETIME}),
        (lambda value: value.month(), {"value": DATETIME}),
        (lambda value: value.day(), {"value": DATETIME}),
        (lambda value: value.hour(), {"value": DATETIME}),
        (lambda value: value.minute(), {"value": DATETIME}),
        (lambda value: value.second(), {"value": DATETIME}),
        (lambda value: value.microsecond(), {"value": DATETIME}),
        (lambda value: value.date(), {"value": DATETIME}),
        (lambda value: value.tzinfo, {"value": DATETIME}),
        (lambda value: value.replace(year=2000), {"value": DATETIME}),
        (lambda value, format: value.strftime(format), {"value": DATETIME, "format": STR}),
        (lambda value: datetime.strptime(value, "%Y %Y"), {"value": STR}),
        (lambda value: datetime.strptime(value, value), {"value": STR}),
        (lambda value: datetime.strptime(value, 1), {"value": STR}),  # pyright: ignore[reportArgumentType]
    ],
)
def test_unsupported_datetime_forms(
    function: Callable[..., object], schema: dict[str, Type]
) -> None:
    with pytest.raises(Unsupported):
        trace(function, schema)


@pytest.mark.parametrize(
    "format",
    [
        "%z",
        "%Z",
        "%c",
        "%x",
        "%X",
        "%I",
        "%p",
        "%U",
        "%W",
        "%V",
        "%G",
        "%u",
        "%w",
        "%b",
        "%a",
        "%B",
        "%A",
        "%e",
        "%Q",
        "%",
        "%-d",
    ],
)
@pytest.mark.parametrize("parse", [False, True])
def test_unsupported_datetime_formats(format: str, parse: bool) -> None:
    function = (
        (lambda value: datetime.strptime(value, format))
        if parse
        else (lambda value: value.strftime(format))
    )
    with pytest.raises(Unsupported, match="datetime directive"):
        trace(function, {"value": STR if parse else DATETIME})
