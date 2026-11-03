import importlib
import json
import random
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pyarrow
import pytest
from pyarrow import parquet

from bench.data import prepare
from bench.runner import ENGINES, run
from bench.timing import Settings
from bench.validation import validate
from tests.benchmark.samples import load
from udfx.frontend import trace
from udfx.types import DATE, FLOAT, INT, STR
from workloads.models import Workload

NAMES = [workload.name for workload in load(Path())]
PARAMETERS = [
    pytest.param(engine, id=engine, marks=pytest.mark.xdist_group(engine)) for engine in ENGINES
]


@pytest.mark.parametrize("engine", PARAMETERS)
@pytest.mark.parametrize("name", NAMES)
def test_sample_inputs_match_python_on_every_engine(
    name: str, engine: str, tmp_path: Path, sample: Path
) -> None:
    workload = next(workload for workload in load(sample) if workload.name == name)
    results = run(workload, [engine], Settings(warmups=0, rounds=1), tmp_path)
    assert len(results) == 2
    for result in results:
        assert result["status"] in {"native", "runtime", "guarded"}, result.get("reason")
        assert len(result["samples"]) == 1
        assert len(result["orders"]) == 1
        assert 0 <= result["native_row_fraction"] <= 1
        assert result["compiled_udfs"] == 1
        assert result["comparisons"]


@pytest.mark.parametrize("engine", PARAMETERS)
def test_benchmark_does_not_time_wrong_results(
    engine: str, tmp_path: Path, sample: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload = load(sample)[0]
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
    engine: str, tmp_path: Path, sample: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload = load(sample)[0]
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


def test_datafusion_baseline_preserves_column_case(tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.table({"source": [1, 2]}), path)
    workload = Workload(
        "test",
        "column_case",
        lambda Value: Value + 1,  # noqa: N803
        {"Value": INT},
        INT,
        path,
        {"Value": "source"},
    )
    results = run(workload, ["datafusion"], Settings(warmups=0, rounds=1), tmp_path / "inputs")
    assert all(result["status"] in {"native", "guarded"} for result in results), results


@pytest.mark.parametrize("engine", PARAMETERS)
def test_nondeterministic_benchmark_reports_python_execution(engine: str, tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.table({"value": [1.0] * 2000, "group": ["a"] * 2000}), path)
    workload = Workload(
        "test",
        "noise",
        lambda value: value + random.gauss(),
        {"value": FLOAT},
        FLOAT,
        path,
        {"value": "value"},
        category="group",
        deterministic=False,
    )
    results = run(workload, [engine], Settings(warmups=0, rounds=1), tmp_path / "inputs")
    assert all(result["status"] == "python" for result in results), results
    assert all(result["validation"] == "distribution" for result in results)
    assert all(result["native_rows"] == result["compiled_udfs"] == 0 for result in results)


@pytest.mark.parametrize("engine", PARAMETERS)
def test_all_fallback_benchmark_reports_python_execution(engine: str, tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.table({"value": ['{"key": "one"}', '{"key": "two"}']}), path)
    workload = Workload(
        "test",
        "objects",
        lambda value: json.dumps(json.loads(value or "{}").get("key")),
        {"value": STR.or_none()},
        STR.or_none(),
        path,
        {"value": "value"},
    )
    results = run(workload, [engine], Settings(warmups=0, rounds=1), tmp_path / "inputs")
    assert all(result["status"] == "python" for result in results), results
    assert all(result["validation"] == "exact" for result in results)
    assert all(result["native_rows"] == result["compiled_udfs"] == 0 for result in results)


@pytest.mark.parametrize("engine", PARAMETERS)
def test_date_workload_preserves_dates_across_baselines(engine: str, tmp_path: Path) -> None:
    path = tmp_path / "dates.parquet"
    values = [date.min, date(1969, 12, 31), date(2000, 2, 28), date(9999, 12, 30), None]
    parquet.write_table(pyarrow.table({"value": pyarrow.array(values, pyarrow.date32())}), path)
    workload = Workload(
        "test",
        "date_shift",
        lambda value: value + timedelta(days=1) if value is not None else None,
        {"value": DATE.or_none()},
        DATE.or_none(),
        path,
        {"value": "value"},
    )
    results = run(workload, [engine], Settings(warmups=0, rounds=1), tmp_path / "inputs")
    assert all(result["status"] in {"native", "guarded"} for result in results), results
    assert all(result["validation"] == "exact" for result in results)
    assert all(result["total_rows"] == len(values) for result in results)
