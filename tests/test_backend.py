import os
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import duckdb
import pytest
from conftest import backend_params, spark_session
from pyspark.sql import SparkSession

from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.duckdb import DuckDBBackend
from udfx.backends.spark import SparkBackend
from udfx.errors import Unsupported
from udfx.ir import FALSE, Col, Lit
from udfx.ops import apply
from udfx.program import Program
from udfx.testing import check_function
from udfx.types import FLOAT, INT, INT64_MAX, INT64_MIN, STR

NO_JAVA = {**os.environ, "JAVA_HOME": "/nonexistent"}
CASE_INSENSITIVE = {"duckdb", "spark"}
FLOATS = [(1.0,), (-1.0,), (0.0,), (-0.0,), (float("inf"),), (float("-inf"),), (float("nan"),)]
YEARS = "SELECT * FROM (VALUES ('2024-01'), ('\uff11\uff12\uff13\uff14')) AS years(text)"


class CustomError(Exception):
    pass


def custom(x: int) -> int:
    raise CustomError(f"custom failure {x}")


def decode(x: int) -> str:
    return bytes([x]).decode()


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


def split(s: str | None) -> list[str] | None:
    return s.split(",") if s else None


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
        backend.run_fallback_rows(Program(inverse, {"x": INT}, Col("x", INT)), [(0,)])
    with pytest.raises(Exception) as error:
        backend.evaluate([backend.column(Col("missing", INT))], {"x": INT}, [(1,)])
    assert not isinstance(error.value, ZeroDivisionError)


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("function", "message"),
    [(custom, "custom failure 255"), (decode, "invalid start byte")],
    ids=["custom", "decode"],
)
def test_fallback_surfaces_exceptions_it_cannot_rebuild(
    backend: Backend[Any], function: Callable[[int], object], message: str
) -> None:
    program = Program(function, {"x": INT}, Col("x", INT))
    expr = backend.fallback(backend.literal(Lit(0, INT)), backend.literal(FALSE), program)
    with pytest.raises(Exception, match=message):
        backend.evaluate([expr], {"x": INT}, [(255,)])


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


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("first", "second", "folding"),
    [("a", "A", {"duckdb", "spark"}), ("\u00e9", "\u00c9", {"spark"})],
    ids=["ascii", "accented"],
)
def test_column_names_that_differ_only_by_case(
    backend: Backend[Any], first: str, second: str, folding: set[str]
) -> None:
    schema = {first: INT, second: INT}
    value = apply("__sub__", Col(first, INT), Col(second, INT))
    program = Program(lambda left, right: left - right, schema, value)
    if backend.name in folding:
        with pytest.raises(Unsupported) as error:
            backend.build(program)
        assert str(error.value) == (
            f"{backend.name} column names are case-insensitive: {first!r} and {second!r} collide"
        )
    else:
        assert backend.evaluate([backend.build(program)], schema, [(5, 3)]) == [[2]]


@pytest.mark.xdist_group("spark")
def test_spark_case_sensitive_session_keeps_case() -> None:
    session = SparkSession.builder.getOrCreate().newSession()
    session.conf.set("spark.sql.caseSensitive", "true")
    assert SparkBackend(session).fold_name("A") == "A"


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("name", ["Amount", "first name", "select", "order", 'a"b', "a.b", "a`b"])
def test_column_names_that_need_quoting(backend: Backend[Any], name: str) -> None:
    program = Program(lambda value: value, {name: INT}, Col(name, INT))
    native = backend.build(program)
    slow = backend.fallback(backend.literal(Lit(0, INT)), backend.literal(FALSE), program)
    assert backend.evaluate([native, slow], {name: INT}, [(5,)]) == [[5], [5]]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("name", ["udfx row", "udfx exact", "udfx out 0", "UDFX ROW"])
def test_reserved_column_names(backend: Backend[Any], name: str) -> None:
    schema = {name: INT}
    program = Program(lambda value: value, schema, Col(name, INT))
    if name.islower() or backend.name in CASE_INSENSITIVE:
        with pytest.raises(Unsupported, match="column names starting with 'udfx ' are reserved"):
            backend.build(program)
        with pytest.raises(Unsupported, match="column names starting with 'udfx ' are reserved"):
            backend.evaluate([backend.literal(Lit(1, INT))], schema, [(5,)])
    else:
        native = backend.build(program)
        slow = backend.fallback(backend.literal(Lit(0, INT)), backend.literal(FALSE), program)
        assert backend.evaluate([native, slow], schema, [(5,), (7,)]) == [[5, 7], [5, 7]]


@pytest.mark.parametrize("backend", backend_params())
def test_nullable_list_literal(backend: Backend[Any]) -> None:
    rows = [("a,b",), ("",), (None,)]
    assert check_function(split, {"s": STR.or_none()}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda x: x if x > 0.0 else -0.0,
        lambda x: x == float("inf"),
        lambda x: x != float("nan"),  # noqa: PLW0177
    ],
    ids=["negative_zero", "infinity", "nan"],
)
def test_float_literals(backend: Backend[Any], function: Callable[[float], object]) -> None:
    assert check_function(function, {"x": FLOAT}, backend, FLOATS) == []


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


@pytest.mark.xdist_group("spark")
def test_spark_keeps_the_session_config() -> None:
    ansi_off, ansi_on = (
        SparkBackend(spark_session(ansi=False)),
        SparkBackend(spark_session(ansi=True)),
    )
    expr = ansi_on.literal(Lit("x", STR)).cast("bigint")
    assert ansi_off.evaluate([expr], {}, [()]) == [[None]]
    with pytest.raises(Exception, match="CAST_INVALID_INPUT"):
        ansi_on.evaluate([expr], {}, [()])


def test_backends_construct_without_an_engine() -> None:
    script = "from udfx.backends import installed; print(*(item.name for item in installed()))"
    result = subprocess.run(
        [sys.executable, "-c", script], env=NO_JAVA, capture_output=True, text=True, check=True
    )
    assert sorted(result.stdout.split()) == ["datafusion", "duckdb", "polars", "spark"]


def test_spark_column_without_java_needs_a_session() -> None:
    script = (
        "from udfx.backends.spark import SparkBackend; from udfx.ir import Col; "
        "from udfx.types import INT; SparkBackend().column(Col('x', INT))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], env=NO_JAVA, capture_output=True, text=True, check=False
    )
    assert "RuntimeError: udfx needs a SparkSession to build Spark columns" in result.stderr
