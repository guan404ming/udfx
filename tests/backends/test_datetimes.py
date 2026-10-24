from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
from conftest import backend_params, spark_session

from udfx import compile
from udfx.backends import arrow
from udfx.backends.base import Backend
from udfx.backends.polars import PolarsBackend
from udfx.backends.spark import SparkBackend
from udfx.ir import FALSE, Col, Lit
from udfx.program import Program
from udfx.testing import check_function
from udfx.testing.strategies import DATETIME_TRAPS
from udfx.types import DATETIME, STR

BOUND = datetime(1970, 1, 1)


def python_datetime(value: datetime) -> datetime:
    if type(value) is not datetime:
        raise TypeError(type(value).__name__)
    return value


@pytest.mark.parametrize("backend", backend_params())
def test_datetime_transport_and_fallback(backend: Backend[Any]) -> None:
    schema = {"value": DATETIME}
    column = Col("value", DATETIME)
    program = Program(python_datetime, schema, column)
    native = backend.column(column)
    patched = backend.fallback(native, backend.literal(FALSE), program)
    values = [*DATETIME_TRAPS, pd.Timestamp("2024-02-29 12:34:56.123456").to_pydatetime()]
    outputs = backend.evaluate(
        [native, patched, backend.literal(Lit(datetime.min, DATETIME))],
        schema,
        [(value,) for value in values],
    )
    assert outputs == [values, values, [datetime.min] * len(values)]
    assert all(type(value) is datetime for column in outputs for value in column)


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("format", "texts"),
    [
        (
            "%Y-%m-%d",
            [
                "2024-02-29",
                "2023-02-29",
                "1900-02-29",
                "2000-02-29",
                "2024-2-9",
                "2024-02-30",
                "0001-01-01",
                "9999-12-31",
                "0000-01-01",
                "2024-01-01junk",
                " 2024-01-01",
                "2024-01-01\n",
                "",
                "2024-00-01",
            ],
        ),
        (
            "%Y-%m-%d %H:%M:%S.%f",
            [
                "2024-02-29 12:34:56.123456",
                "1969-12-31 23:59:59.999999",
                "0001-01-01 00:00:00.000000",
                "2024-01-01 00:00:00.1",
                "2024-01-01 24:00:00.000000",
                "2024-01-01 00:00:60.000000",
                "9999-12-31 23:59:59.999999",
            ],
        ),
        ("%d/%m/%y", ["01/02/68", "01/02/69", "31/12/00", "1/2/24", "29/02/23"]),
        ("%Y %j", ["2024 060", "2023 366", "2024 367", "0001 001"]),
        ("%H:%M", ["23:59", "00:00", "1:2", "24:00"]),
        ("day %% %d.%m.%Y", ["day % 29.02.2024", "day % 1.1.1970"]),
    ],
)
def test_strptime_exactness_and_fallback(
    backend: Backend[Any], format: str, texts: list[str]
) -> None:
    def function(text: str) -> datetime:
        return datetime.strptime(text, format)

    assert (
        check_function(
            function, {"text": STR.or_none()}, backend, [(text,) for text in [*texts, None]]
        )
        == []
    )


@pytest.mark.parametrize("backend", backend_params())
def test_strptime_guard(backend: Backend[Any]) -> None:
    translation = compile(lambda text: datetime.strptime(text, "%d/%m/%Y"), {"text": STR}, backend)
    assert translation.program is not None
    lowered = backend.lower(translation.program.value)
    assert lowered.exact is not None
    rows = [("29/02/2024",), ("9/2/2024",), ("29/02/2023",), ("01/01/2024junk",)]
    flags = backend.evaluate([lowered.exact], {"text": STR}, rows)[0]
    assert flags[0] is True
    assert all(flag is not True for flag in flags[1:])


@pytest.mark.xdist_group("spark")
def test_spark_strptime_skips_render() -> None:
    backend = SparkBackend()
    translation = compile(lambda text: datetime.strptime(text, "%Y-%m-%d"), {"text": STR}, backend)
    assert translation.program is not None
    assert "lpad" not in str(backend.lower(translation.program.value).expr)


@pytest.mark.parametrize("backend", backend_params())
def test_strptime_catches_value_error(backend: Backend[Any]) -> None:
    def year(text: str) -> int:
        try:
            return datetime.strptime(text, "%Y-%m-%d").year
        except ValueError:
            return -1

    assert check_function(year, {"text": STR}, backend, [("2024-02-29",), ("2023-02-29",)]) == []


@pytest.mark.xdist_group("spark")
@pytest.mark.parametrize("timezone", ["UTC", "America/Los_Angeles", "Asia/Taipei"])
@pytest.mark.parametrize("ansi", [False, True])
def test_spark_datetime_timezone(timezone: str, ansi: bool) -> None:
    session = spark_session(ansi).newSession()
    session.conf.set("spark.sql.session.timeZone", timezone)
    backend = SparkBackend(session)
    values = DATETIME_TRAPS
    schema = {"value": DATETIME}
    rows = [(value,) for value in values]
    functions = [
        lambda value: value.hour,
        lambda value: value.strftime("%Y-%m-%d %H:%M:%S.%f"),
        lambda value: value == BOUND,
        lambda value: (value - BOUND).days,
        lambda value: (BOUND - value).seconds,
        lambda value: value + timedelta(microseconds=1),
        lambda value: value - timedelta(days=1),
    ]
    for function in functions:
        assert check_function(function, schema, backend, rows) == []
    assert (
        check_function(
            lambda text: datetime.strptime(text, "%Y-%m-%d %H:%M"),
            {"text": STR},
            backend,
            [("2024-03-10 02:30",), ("2024-11-03 01:30",), ("2024-3-10 2:30",)],
        )
        == []
    )


def test_arrow_pandas_datetime_fallback() -> None:
    value = pd.Timestamp("2024-02-29 12:34:56.123456")
    column = pa.array([value], type=pa.timestamp("ns"))
    program = Program(python_datetime, {"value": DATETIME}, Col("value", DATETIME))
    native = pa.array([None], type=arrow.dtype(DATETIME))
    output = arrow.fallback(PolarsBackend(), program, pa.array([False]), native, [column])
    assert output.to_pylist() == [value.to_pydatetime()]
    assert type(output[0].as_py()) is datetime
