from collections.abc import Callable
from typing import Any

import duckdb
import pytest
from conftest import backend_params

from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.duckdb import DuckDBBackend
from udfx.ir import FALSE, Col, Lit
from udfx.program import Program
from udfx.testing import check_function
from udfx.types import INT, INT64_MAX, INT64_MIN, STR

YEARS = "SELECT * FROM (VALUES ('2024-01'), ('\uff11\uff12\uff13\uff14')) AS years(text)"


def inverse(x: int) -> float:
    return 1 / x


def year(text: str) -> int:
    return int(text[:4]) + 1


def below_int64(x: int) -> bool:
    return x < 2**63


def above_int64(x: int) -> bool:
    return x > -(2**63) - 1


def constant_branch(value: int) -> int:
    if value == 0:
        return 1 // 0
    return 2


@pytest.mark.parametrize("backend", backend_params())
def test_evaluate_keeps_row_count_without_columns(backend: Backend[Any]) -> None:
    program = Program(lambda: 1, {}, Lit(1, INT))
    assert backend.evaluate([backend.build(program)], {}, [(), (), ()]) == [[1, 1, 1]]


@pytest.mark.parametrize("backend", backend_params())
def test_fallback_raises_the_python_exception(backend: Backend[Any]) -> None:
    assert check_function(inverse, {"x": INT}, backend, [(2,), (0,), (4,)]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_evaluate_forgets_earlier_python_exception(backend: Backend[Any]) -> None:
    with pytest.raises((ZeroDivisionError, RuntimeError)):
        backend.patch(Program(inverse, {"x": INT}, Col("x", INT)), [(0,)])
    with pytest.raises(Exception) as error:
        backend.evaluate([backend.column(Col("missing", INT))], {"x": INT}, [(1,)])
    assert not isinstance(error.value, ZeroDivisionError)


@pytest.mark.parametrize("backend", backend_params())
def test_fallback_patches_inexact_rows(backend: Backend[Any]) -> None:
    rows = [("2024-01",), (" 123",), ("\uff11\uff12\uff13\uff14",), ("9223372036854775807",)]
    assert check_function(year, {"text": STR}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_fallback_patches_rows_without_columns(backend: Backend[Any]) -> None:
    assert check_function(lambda: int("\uff11\uff12"), {}, backend, [(), ()]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_fallback_broadcasts_constant_inputs(backend: Backend[Any]) -> None:
    assert check_function(constant_branch, {"value": INT}, backend, [(1,), (1,), (0,)]) == []
    program = Program(lambda value: value + 1, {"value": INT}, Col("value", INT))
    expr = backend.fallback(backend.literal(Lit(0, INT)), backend.literal(FALSE), program)
    assert backend.evaluate([expr], {"value": INT}, [(1,), (2,)]) == [[2, 3]]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [below_int64, above_int64], ids=lambda f: f.__name__)
def test_fallback_runs_literals_outside_int64(
    backend: Backend[Any], function: Callable[[int], bool]
) -> None:
    rows = [(0,), (INT64_MAX,), (INT64_MIN,)]
    assert check_function(function, {"x": INT}, backend, rows) == []


@pytest.mark.xdist_group("duckdb")
def test_duckdb_string_form_runs_on_default_connection() -> None:
    translation = compile(year, {"text": STR}, "duckdb")
    assert translation.ok
    assert duckdb.sql(YEARS).select(translation.expr).fetchall() == [(2025,), (1235,)]


@pytest.mark.xdist_group("duckdb")
def test_duckdb_explicit_connection_registers_each_program_once() -> None:
    connection = duckdb.connect()
    backend = DuckDBBackend(connection)
    translation = compile(year, {"text": STR}, backend)
    assert translation.program is not None
    again = backend.build(translation.program)
    for expr in (translation.expr, again):
        assert connection.sql(YEARS).select(expr).fetchall() == [(2025,), (1235,)]
