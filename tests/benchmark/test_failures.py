from pathlib import Path

import polars
import pytest

from bench.engines.polars import Engine
from bench.runner import run
from bench.timing import Settings
from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from workloads.flights import load
from workloads.storage import FIXTURES


@pytest.mark.parametrize("error", [Unsupported("no rule"), RuntimeError("engine failed")])
def test_lowering_failure_keeps_all_results_and_closes_engine(
    error: Exception, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed = []

    def fail(*args: object) -> None:
        raise error

    monkeypatch.setattr(PolarsBackend, "build", fail)
    monkeypatch.setattr(Engine, "close", lambda self: closed.append(True))
    results = run(load(FIXTURES)[0], ["polars"], Settings(rounds=1, warmups=0), tmp_path)
    assert closed == [True]
    assert len(results) == 2
    assert {result["status"] for result in results} == {
        "unsupported" if isinstance(error, Unsupported) else "error"
    }


def test_missing_diagnostics_cannot_claim_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(PolarsBackend, "build", lambda *args: polars.lit("value"))
    results = run(load(FIXTURES)[0], ["polars"], Settings(rounds=1, warmups=0), tmp_path)
    assert all(
        result["status"] == "error" and "fallback guard" in result["reason"] for result in results
    )


def test_cleanup_failure_does_not_abort_the_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object) -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(Engine, "close", fail)
    results = run(load(FIXTURES)[0], ["polars"], Settings(rounds=1, warmups=0), tmp_path)
    assert all(
        result["status"] == "error" and "cleanup failed" in result["reason"] for result in results
    )
