from dataclasses import replace
from typing import Any

import duckdb
import polars
import pytest
from conftest import backend_params

from bench.fingerprint import expression_text, fingerprint
from udfx.backends.base import Backend
from udfx.backends.datafusion import DataFusionBackend
from udfx.backends.duckdb import DuckDBBackend
from udfx.frontend import trace
from udfx.ir import Col
from udfx.program import Program
from udfx.types import BOOL, INT, STR
from workloads.nyc311 import postal_code


@pytest.mark.parametrize("backend", backend_params())
def test_fingerprint_groups_lowered_expressions(backend: Backend[Any]) -> None:
    programs = [
        trace(lambda value: isinstance(value, str), {"value": STR.or_none()}),
        trace(lambda text: text is not None, {"text": STR.or_none()}),
        trace(lambda text: text is None, {"text": STR.or_none()}),
    ]
    fingerprints = [fingerprint(backend, program) for program in programs]
    assert fingerprints[0] == fingerprints[1]
    assert fingerprints[0] != fingerprints[2]
    assert "c0" in fingerprints[0]


@pytest.mark.parametrize("backend", backend_params())
def test_fingerprint_preserves_literals(backend: Backend[Any]) -> None:
    programs = [
        trace(lambda value: value == "value", {"value": STR}),
        trace(lambda text: text == "value", {"text": STR}),
        trace(lambda text: text == "text", {"text": STR}),
    ]
    fingerprints = [fingerprint(backend, program) for program in programs]
    assert fingerprints[0] == fingerprints[1]
    assert fingerprints[0] != fingerprints[2]


@pytest.mark.parametrize("backend", backend_params())
def test_fingerprint_uses_schema_positions(backend: Backend[Any]) -> None:
    programs = [
        trace(lambda row: row["c1"] > row["c0"], {"c1": INT, "c0": INT}),
        trace(lambda first, second: first > second, {"first": INT, "second": INT}),
        trace(lambda first, second: second > first, {"first": INT, "second": INT}),
    ]
    fingerprints = [fingerprint(backend, program) for program in programs]
    assert fingerprints[0] == fingerprints[1]
    assert fingerprints[0] != fingerprints[2]
    assert "c0" in fingerprints[0]
    assert "c1" in fingerprints[0]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("title", [False, True])
def test_fingerprint_normalizes_quoted_columns(backend: Backend[Any], title: bool) -> None:
    def program(name: str) -> Program:
        return trace(lambda row: row[name].title() if title else row[name] == "value", {name: STR})

    fingerprints = {
        fingerprint(backend, program(name))
        for name in ("value", "two words", 'a"b', "a.b", "lower")
    }
    assert len(fingerprints) == 1


@pytest.mark.parametrize("backend", backend_params())
def test_fingerprint_exposes_fallback_native(backend: Backend[Any]) -> None:
    programs = [
        trace(lambda value: value.capitalize(), {"value": STR}),
        trace(lambda text: text.capitalize(), {"text": STR}),
        trace(lambda value: value.swapcase(), {"value": STR}),
    ]
    original = expression_text(backend.column(Col("value", STR)), backend.name)
    fingerprints = [fingerprint(backend, program) for program in programs]
    assert fingerprints[0] == fingerprints[1]
    assert fingerprints[0] != fingerprints[2]
    assert all(" fallback when not " in value for value in fingerprints)
    assert expression_text(backend.column(Col("value", STR)), backend.name) == original


@pytest.mark.parametrize("backend", backend_params())
def test_fingerprint_exposes_fault_guards(backend: Backend[Any]) -> None:
    program = Program(
        lambda row: row["value"],
        {"value": STR, "first": BOOL, "second": BOOL},
        Col("value", STR),
        Col("first", BOOL),
        row_mode=True,
    )
    first = fingerprint(backend, program)
    second = fingerprint(backend, replace(program, fault=Col("second", BOOL)))
    assert first != second
    if backend.name in {"polars", "datafusion"}:
        assert (
            first.partition(" fallback when not ")[0] == second.partition(" fallback when not ")[0]
        )
        assert " fallback when not " in first
    else:
        assert "udfx fault fingerprint:" in first
        assert " fallback when not " not in first


@pytest.mark.parametrize("backend", backend_params())
def test_fingerprint_normalizes_overflow_programs(backend: Backend[Any]) -> None:
    programs = [
        trace(lambda value: value + 1, {"value": INT}),
        trace(lambda number: number + 1, {"number": INT}),
        trace(lambda value: value + 2, {"value": INT}),
    ]
    fingerprints = [fingerprint(backend, program) for program in programs]
    assert fingerprints[0] == fingerprints[1]
    assert fingerprints[0] != fingerprints[2]


def test_polars_fingerprint_keeps_long_literals() -> None:
    values = list(range(20))
    first = polars.lit(values)
    values[10] = 42
    second = polars.lit(values)
    assert str(first) == str(second)
    assert expression_text(first, "polars") != expression_text(second, "polars")


def test_duckdb_fingerprint_strips_fallback_identity() -> None:
    name = f"udfx fallback {'a' * 64} 17"
    literal = duckdb.ConstantExpression(f"quoted '{name}' and \"{name}\"(")
    first = duckdb.FunctionExpression(name, literal)
    second = duckdb.FunctionExpression(f"udfx fallback {'b' * 64} 23", literal)
    assert expression_text(first, "duckdb") == expression_text(second, "duckdb")
    assert str(literal) in expression_text(first, "duckdb")


def test_duckdb_fingerprint_normalizes_generated_bindings() -> None:
    backend = DuckDBBackend()
    program = trace(postal_code, {"value": STR.or_none()})
    assert fingerprint(backend, program) == fingerprint(backend, program)
    literal = duckdb.ConstantExpression('literal "udfx element 17"')
    expressions = [
        duckdb.LambdaExpression(name, duckdb.ColumnExpression(name) + literal)
        for name in ("udfx element 17", "udfx element 23")
    ]
    first, second = [expression_text(expr, "duckdb") for expr in expressions]
    assert first == second
    assert str(literal) in first


def test_datafusion_fingerprint_keeps_stable_binding_names() -> None:
    backend = DataFusionBackend()
    literal = 'literal "udfx element 17"'
    program = trace(lambda text: [part + literal for part in text.split()], {"text": STR})
    first = fingerprint(backend, program)
    assert first == fingerprint(backend, program)
    assert literal in first
