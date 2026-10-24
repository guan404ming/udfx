import re
from typing import Any

import pytest
from conftest import backend_params

from udfx import explain
from udfx.backends.base import Backend
from udfx.program import Program
from udfx.types import FLOAT, INT, STR


def text_length(text: str) -> int:
    return len(text)


def float_text(value: float) -> str:
    return str(value)


@pytest.mark.parametrize("backend", backend_params())
def test_explain_native_and_guarded_plans(backend: Backend[Any]) -> None:
    native = explain(text_length, {"text": STR}, backend)
    assert f"Backend: {backend.name}" in native
    assert "Status: native" in native
    assert "Python fallback: no" in native
    assert "Fault handling: none" in native
    assert "Native guard:" not in native
    assert re.search(r"Compile time: \d+\.\d{6} s", native)
    guarded = explain(float_text, {"value": FLOAT}, backend)
    assert "Status: guarded" in guarded
    assert "Python fallback: possible" in guarded
    assert "Fallback rows: not measured" in guarded
    assert "Native guard:" in guarded
    assert "Reason: native values need a Python-equivalence check" in guarded


@pytest.mark.parametrize("backend", backend_params())
def test_explain_fault_handling(backend: Backend[Any]) -> None:
    report = explain(lambda value: value + 1, {"value": INT}, backend)
    if backend.name in {"spark", "duckdb"}:
        assert "Status: native" in report
        assert "Python fallback: no" in report
        assert "Fault handling: native errors" in report
    else:
        assert "Status: guarded" in report
        assert "Fault handling: Python fallback" in report
        assert "Reason: Python exceptions or integer overflow need Python execution" in report


@pytest.mark.parametrize("backend", backend_params())
def test_explain_builds_once_without_running_rows(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = backend.build
    builds = []

    def build(program: Program) -> Any:
        builds.append(program)
        return original(program)

    def run(*args: object) -> None:
        raise AssertionError("explain must not execute input rows")

    monkeypatch.setattr(backend, "build", build)
    monkeypatch.setattr(Program, "run", run)
    explain(float_text, {"value": FLOAT}, backend)
    assert len(builds) == 1
