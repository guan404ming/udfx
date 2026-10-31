import json
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import pytest
from conftest import spark_session

from udfx import explain
from udfx.backends.diagnostics import capture_lowering
from udfx.backends.lowering_context import lowering_context, record_runtime, runtime_udfs
from udfx.backends.program_lowering import lower_program
from udfx.backends.spark import SparkBackend
from udfx.backends.spark import backend as spark_backend
from udfx.backends.spark.bindings import CURRENT
from udfx.frontend import trace
from udfx.ir import FALSE
from udfx.types import FLOAT, INT, STR


def packed(text: str) -> str:
    return json.dumps(text.split())


def guarded(text: str, value: float) -> str:
    return json.dumps(text.split()) + str(value)


def length(text: str) -> int:
    return len(text)


@pytest.mark.parametrize("failed", [False, True])
def test_runtime_context_tracks_successful_children(failed: bool) -> None:
    record_runtime("outside")
    with lowering_context():
        record_runtime("outer")
        with (
            pytest.raises(ValueError) if failed else nullcontext(),
            lowering_context(binds=False),
        ):
            record_runtime("child")
            record_runtime("child")
            if failed:
                raise ValueError("abandoned lowering")
        assert runtime_udfs() == (("outer",) if failed else ("child", "outer"))
    with lowering_context():
        assert runtime_udfs() == ()


@pytest.mark.xdist_group("spark")
@pytest.mark.parametrize("function", [packed, guarded])
def test_explain_reports_actual_spark_runtime(function: Any) -> None:
    backend = SparkBackend(spark_session(ansi=True))
    schema = {"text": STR} if function is packed else {"text": STR, "value": FLOAT}
    report = explain(function, schema, backend)
    assert "Runtime UDFs: spark.json_words" in report
    assert "Status: runtime" in report if function is packed else "Status: guarded" in report
    assert (
        "Python fallback: no" in report
        if function is packed
        else "Python fallback: possible" in report
    )
    assert "Expression:" in report
    plain = explain(length, {"text": STR}, backend)
    assert "Status: native" in plain
    assert "Runtime UDFs:" not in plain


@pytest.mark.xdist_group("spark")
def test_spark_retry_discards_first_runtime_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = SparkBackend(spark_session(ansi=True))
    calls = []

    def lower(*arguments: Any) -> Any:
        calls.append(True)
        with lowering_context() as context:
            record_runtime("discarded" if len(calls) == 1 else "retained")
            lowered = lower_program(*arguments)
            if len(calls) == 1:
                context.has_bindings = True
                scope = CURRENT.get()
                assert scope is not None
                scope.requires_unbound = True
                return replace(lowered, exact=backend.literal(FALSE))
            return lowered

    monkeypatch.setattr(spark_backend, "lower_program", lower)
    with capture_lowering() as details:
        backend.lower_program(trace(lambda value: value, {"value": INT}))
    assert len(calls) == 2
    assert len(details) == 1
    assert details[0].runtime_udfs == ("retained",)
