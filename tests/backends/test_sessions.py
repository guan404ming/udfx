import gc
import importlib
import os
import subprocess
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import duckdb
import pytest
from conftest import backend_params, spark_session
from pyspark.sql import SparkSession

from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.duckdb import DuckDBBackend
from udfx.backends.spark import SparkBackend
from udfx.ir import Lit
from udfx.testing import check_function
from udfx.types import INT, STR

from .functions import inverse

NO_JAVA = {**os.environ, "JAVA_HOME": "/nonexistent"}
YEARS = "SELECT * FROM (VALUES ('2024-01'), ('\uff11\uff12\uff13\uff14')) AS years(text)"
ZERO = "SELECT 0::BIGINT AS x"
FALLBACKS = "SELECT count(*) FROM duckdb_functions() WHERE function_name LIKE 'udfx fallback %'"


def guarded_year(text: str) -> int:
    return (int(text[:4]) + 2) - 1


def exclaim(s: str) -> str:
    return s.lower() + "!"


@pytest.mark.parametrize("backend", backend_params())
def test_case_drift_probe_runs_on_the_engine(
    backend: Backend[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UDFX_CACHE_DIR", str(tmp_path))
    strings = importlib.import_module(f"udfx.backends.{backend.name}.strings")
    assert strings.native_recase("lower", ["A", "b", "C"]) == ["a", "b", "c"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.xdist_group("duckdb")
def test_duckdb_fallback_releases_the_connection() -> None:
    connection = duckdb.connect()
    reference = weakref.ref(connection)
    translation = compile(guarded_year, {"text": STR}, DuckDBBackend(connection))
    assert translation.ok
    del connection
    gc.collect()
    assert reference() is None


@pytest.mark.xdist_group("duckdb")
def test_duckdb_string_form_runs_on_default_connection() -> None:
    translation = compile(guarded_year, {"text": STR}, "duckdb")
    assert translation.ok
    assert duckdb.sql(YEARS).select(translation.expr).fetchall() == [(2025,), (1235,)]


@pytest.mark.xdist_group("duckdb")
def test_duckdb_explicit_connection_registers_each_program_once() -> None:
    connection = duckdb.connect()
    backend = DuckDBBackend(connection)
    translation = compile(guarded_year, {"text": STR}, backend)
    assert translation.program is not None
    again = backend.build(translation.program)
    for expr in (translation.expr, again):
        assert connection.sql(YEARS).select(expr).fetchall() == [(2025,), (1235,)]


@pytest.mark.xdist_group("duckdb")
def test_duckdb_recompiling_reuses_the_fallback_function() -> None:
    connection = duckdb.connect()
    for _ in range(3):
        assert compile(guarded_year, {"text": STR}, DuckDBBackend(connection)).ok
    assert connection.sql(FALLBACKS).fetchall() == [(1,)]
    expr = compile(inverse, {"x": INT}, DuckDBBackend(connection)).expr
    with pytest.raises(ZeroDivisionError):
        DuckDBBackend(connection).execute(lambda: connection.sql(ZERO).select(expr).fetchall())


def run_years(connection: duckdb.DuckDBPyConnection) -> list[tuple[object, ...]]:
    expr = compile(guarded_year, {"text": STR}, DuckDBBackend(connection)).expr
    return connection.sql(YEARS).select(expr).fetchall()


@pytest.mark.xdist_group("duckdb")
def test_duckdb_registers_on_each_connection_of_one_database(tmp_path: Path) -> None:
    same = duckdb.connect()
    parent = duckdb.connect()
    path = str(tmp_path / "years.db")
    pairs = [(same, same), (parent.cursor(), parent), (duckdb.connect(path), duckdb.connect(path))]
    for first, second in pairs:
        assert run_years(first) == run_years(second) == [(2025,), (1235,)]
    assert same.sql(FALLBACKS).fetchall() == [(1,)]


@pytest.mark.xdist_group("duckdb")
def test_duckdb_threads_with_their_own_cursors() -> None:
    connection = duckdb.connect()
    barrier = threading.Barrier(8)

    def work(cursor: duckdb.DuckDBPyConnection) -> list[tuple[object, ...]]:
        barrier.wait()
        return run_years(cursor)

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(work, [connection.cursor() for _ in range(8)]))
    assert results == [[(2025,), (1235,)]] * 8


@pytest.mark.xdist_group("spark")
def test_spark_explicit_session_runs_fallback_rows() -> None:
    backend = SparkBackend(SparkSession.getActiveSession() or SparkSession.builder.getOrCreate())
    assert check_function(exclaim, {"s": STR}, backend, [("İ",), ("a",)]) == []


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


@pytest.mark.xdist_group("spark")
def test_spark_rejects_a_stopped_session() -> None:
    script = (
        "from pyspark.sql import SparkSession; from udfx.backends.spark import SparkBackend; "
        "session = SparkSession.builder.getOrCreate(); session.stop(); "
        "SparkBackend(session).session()"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert "RuntimeError: the SparkSession given to SparkBackend was stopped" in result.stderr
