# ruff: noqa: PLW0108

from __future__ import annotations

import datetime as datetime_module
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from itertools import product
from typing import Any

import pytest
from conftest import frontend_backend_params

from udfx.backends.base import Backend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import Call, Lit
from udfx.testing import check_function
from udfx.testing.datetime_traps import DATE_TRAPS, DATETIME_TRAPS
from udfx.types import DATE, DATETIME, FLOAT, INT, STR, Type, accepts, fits, type_of, unify

PARSE = date.fromisoformat
BOUND = date(2000, 2, 29)


def test_date_type_identity() -> None:
    assert type_of(BOUND) == DATE
    assert fits(BOUND, DATE)
    assert not fits(BOUND, DATETIME)
    assert type_of(datetime.min) == DATETIME
    assert not fits(datetime.min, DATE)
    aware = datetime(2000, 1, 1, tzinfo=UTC)
    assert type_of(aware) is None
    assert not fits(aware, DATE)
    assert not fits(aware, DATETIME)
    assert not accepts(DATE, DATETIME)
    assert not accepts(DATETIME, DATE)
    assert unify(DATE, DATETIME) is None


@pytest.mark.parametrize(
    "function",
    [
        lambda: date(2000, 2, 29),
        lambda: date(year=2000, month=2, day=29),
        lambda: datetime_module.date(2000, 2, 29),
        lambda: date.fromisoformat("2000-02-29"),
        lambda: PARSE("2000-02-29"),
        lambda: date.fromordinal(730179),
        lambda: BOUND,
        lambda: date.min,
        lambda: date.max,
    ],
)
def test_literal_date_forms(function: Callable[..., object]) -> None:
    program = trace(function, {})
    assert program.result == DATE
    assert isinstance(program.value, Lit)
    assert program.value.value == function()


@pytest.mark.parametrize(
    ("function", "name"),
    [
        (lambda value: value.year, ".year"),
        (lambda value: value.month, ".month"),
        (lambda value: value.day, ".day"),
        (lambda value: value.weekday(), ".weekday"),
        (lambda value: value.isoweekday(), ".isoweekday"),
        (lambda value: value.toordinal(), ".toordinal"),
    ],
)
def test_date_attribute_forms(function: Callable[..., object], name: str) -> None:
    program = trace(function, {"value": DATE})
    assert isinstance(program.value, Call)
    assert program.value.op.name == name


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: value + timedelta(hours=12),
        lambda value: value - timedelta(hours=12),
        lambda value: value + timedelta(hours=-12),
        lambda value: value - timedelta(hours=-12),
        lambda value: timedelta(microseconds=-1) + value,
        lambda value: value - timedelta(seconds=-1),
        lambda value: value + date.resolution,
        lambda value: value + timedelta(days=999999999),
        lambda value: value - timedelta(days=999999999),
    ],
)
def test_date_duration_normalization(
    function: Callable[..., object], backend: Backend[Any]
) -> None:
    rows = [(value,) for value in (*DATE_TRAPS, None)]
    assert check_function(function, {"value": DATE.or_none()}, backend, rows) == []


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize(
    ("function", "result"),
    [
        (lambda left, right: (left - right).days, INT),
        (lambda left, right: (left - right).seconds, INT),
        (lambda left, right: (left - right).total_seconds(), FLOAT),
    ],
)
def test_date_duration_attributes(
    function: Callable[..., object], result: Type, backend: Backend[Any]
) -> None:
    schema = {"left": DATE.or_none(), "right": DATE.or_none()}
    assert trace(function, schema).result == result
    assert (
        check_function(function, schema, backend, list(product((*DATE_TRAPS, None), repeat=2)))
        == []
    )


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda left, right: left == right,
        lambda left, right: left != right,
        lambda left, right: left < right if left is not None else False,
        lambda left, right: right >= left if left is not None else False,
        lambda left, right: (left - right).days if left is not None else 0,
        lambda left, right: (right - left).days if left is not None else 0,
    ],
)
def test_mixed_date_datetime(function: Callable[..., object], backend: Backend[Any]) -> None:
    schema = {"left": DATE.or_none(), "right": DATETIME.or_none()}
    rows = list(product((*DATE_TRAPS, None), (*DATETIME_TRAPS, None)))
    assert check_function(function, schema, backend, rows) == []


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize(
    "function", [lambda value: value.date(), lambda value: datetime.date(value)]
)
def test_datetime_to_date(function: Callable[..., object], backend: Backend[Any]) -> None:
    schema = {"value": DATETIME.or_none()}
    program = trace(function, schema)
    assert program.result == DATE
    assert isinstance(program.value, Call)
    assert program.value.op.name == ".date"
    assert (
        check_function(function, schema, backend, [(value,) for value in (*DATETIME_TRAPS, None)])
        == []
    )


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize(
    "function", [lambda value: isinstance(value, date), lambda value: isinstance(value, datetime)]
)
@pytest.mark.parametrize(("type", "value"), [(DATE, BOUND), (DATETIME, datetime.min)])
def test_date_type_checks(
    function: Callable[..., object], type: Type, value: object, backend: Backend[Any]
) -> None:
    assert check_function(function, {"value": type.or_none()}, backend, [(value,), (None,)]) == []


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: date(2001, 2, 29) if value else BOUND,
        lambda value: date.fromordinal(0) if value else BOUND,
        lambda value: date.fromisoformat("invalid") if value else BOUND,
    ],
)
def test_date_constructor_faults(function: Callable[..., object], backend: Backend[Any]) -> None:
    assert check_function(function, {"value": INT}, backend, [(0,), (1,)]) == []


@pytest.mark.parametrize(
    ("function", "schema"),
    [
        (lambda value: date(value, 1, 1), {"value": INT}),
        (lambda value: date.fromisoformat(value), {"value": STR}),
        (lambda value: date.fromordinal(value), {"value": INT}),
        (lambda value: value.year(), {"value": DATE}),
        (lambda value: value.hour, {"value": DATE}),
    ],
)
def test_unsupported_date_forms(function: Callable[..., object], schema: dict[str, Type]) -> None:
    with pytest.raises(Unsupported):
        trace(function, schema)


class SpoofedDateParser:
    __self__ = date
    __name__ = "fromisoformat"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> date:
        self.calls += 1
        return BOUND


def test_date_parser_requires_builtin_method() -> None:
    parser = SpoofedDateParser()

    def function() -> date:
        return parser("2000-02-29")

    with pytest.raises(Unsupported, match="unknown callable"):
        trace(function, {})
    assert parser.calls == 0
