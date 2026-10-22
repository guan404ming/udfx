from typing import Any

import pytest
from conftest import backend_params

from bench.speed.timing import fallback_in_plan, pending_share
from udfx.backends.base import Backend
from udfx.backends.duckdb import DuckDBBackend
from udfx.frontend import trace
from udfx.ir import Col, Guarded
from udfx.program import Program
from udfx.types import BOOL, FLOAT, INT, STR


@pytest.mark.parametrize("backend", backend_params())
def test_plan_tracks_fallback_after_native_faults(backend: Backend[Any]) -> None:
    program = trace(lambda value: 1.0 / value, {"value": FLOAT})
    expr = backend.build(program)
    assert fallback_in_plan(expr, backend.name) == (backend.name in {"polars", "datafusion"})
    rows = [(0.0,), (1.0,)]
    share = pending_share(
        backend, program, lambda exact: backend.evaluate([exact], program.schema, rows)[0]
    )
    assert share == (0.5 if backend.name in {"polars", "datafusion"} else 0.0)


@pytest.mark.parametrize("backend", backend_params())
def test_pending_share_counts_false_and_null(backend: Backend[Any]) -> None:
    schema = {"value": INT, "flag": BOOL.or_none()}
    program = Program(
        lambda value, flag: value,
        schema,
        Guarded(Col("value", INT), Col("flag", BOOL.or_none())),
    )
    assert fallback_in_plan(backend.build(program), backend.name)
    rows = [(1, True), (2, False), (3, None), (4, True)]
    assert (
        pending_share(backend, program, lambda exact: backend.evaluate([exact], schema, rows)[0])
        == 0.5
    )


@pytest.mark.parametrize("backend", backend_params())
def test_exact_plan_has_zero_pending_share(backend: Backend[Any]) -> None:
    program = trace(lambda value: value == 0, {"value": FLOAT})
    assert not fallback_in_plan(backend.build(program), backend.name)
    assert pending_share(backend, program, lambda exact: pytest.fail("unneeded query")) == 0.0


def test_python_programs_are_fully_pending() -> None:
    backend = DuckDBBackend()
    program = trace(lambda left, right: len(set(left) & set(right)), {"left": STR, "right": STR})
    assert fallback_in_plan(backend.build(program), backend.name)
    assert pending_share(backend, program, lambda exact: []) == 1.0
