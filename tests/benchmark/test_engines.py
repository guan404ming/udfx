import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow
import pytest
from pyarrow import parquet

from bench.data import prepare
from bench.runner import ENGINES, run
from bench.timing import Settings
from bench.validation import validate
from udfx.frontend import trace
from udfx.types import STR
from workloads.catalog import load
from workloads.models import Workload
from workloads.storage import FIXTURES

WORKLOADS = load(["flights", "nyc311"], FIXTURES)
PARAMETERS = [
    pytest.param(engine, id=engine, marks=pytest.mark.xdist_group(engine)) for engine in ENGINES
]


@pytest.mark.parametrize("engine", PARAMETERS)
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda workload: workload.identifier)
def test_public_inputs_match_python_on_every_engine(
    workload: Workload, engine: str, tmp_path: Path
) -> None:
    results = run(workload, [engine], Settings(warmups=0, rounds=1), tmp_path)
    assert len(results) == 2
    for result in results:
        assert result["status"] in {"native", "guarded"}, result.get("reason")
        assert len(result["samples"]) == 1
        assert len(result["orders"]) == 1
        assert 0 <= result["native_row_fraction"] <= 1
        assert result["compiled_udfs"] == 1
        assert result["comparisons"]


@pytest.mark.parametrize("engine", PARAMETERS)
def test_benchmark_does_not_time_wrong_results(
    engine: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload = WORKLOADS[2]
    module = importlib.import_module(f"bench.engines.{engine}")
    original = module.Engine.python

    def wrong(self: Any) -> dict:
        self.workload = replace(self.workload, function=lambda origin, destination: "wrong")
        return original(self)

    monkeypatch.setattr(module.Engine, "python", wrong)
    monkeypatch.setattr("bench.runner.paired", lambda *args: pytest.fail("Timed wrong output"))
    results = run(workload, [engine], Settings(warmups=0, rounds=1), tmp_path)
    assert all(
        result["status"] == "error" and "AssertionError" in result["reason"] for result in results
    )
    assert all("samples" not in result for result in results)


@pytest.mark.parametrize("engine", PARAMETERS)
def test_baselines_do_not_call_program_run(
    engine: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload = WORKLOADS[2]
    data = prepare(workload, tmp_path, 3)
    assert workload.function is not None
    program = trace(workload.function, workload.schema)
    monkeypatch.setattr(type(program), "run", lambda *args: pytest.fail("Baseline used udfx"))
    adapter = importlib.import_module(f"bench.engines.{engine}").Engine(workload, data, 1)
    try:
        for expr in adapter.python().values():
            validate(adapter.execute(expr, "memory"), data, "memory", engine)
    finally:
        adapter.close()


@pytest.mark.parametrize("engine", PARAMETERS)
def test_input_names_cannot_collide_with_harness_columns(engine: str, tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    parquet.write_table(
        pyarrow.table({"first": ["a", "b"], "second": ["c", "d"], "group": ["one", "two"]}), path
    )
    workload = Workload(
        "test",
        "names",
        lambda position, category: position + category,
        {"position": STR, "category": STR},
        STR,
        path,
        {"position": "first", "category": "second"},
        "group",
    )
    results = run(workload, [engine], Settings(warmups=0, rounds=1), tmp_path / "inputs")
    assert all(result["status"] == "native" for result in results), results
