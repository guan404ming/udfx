import csv
import json
from pathlib import Path

import pyarrow
import pytest
from pyarrow import parquet

from bench.report import export_csv, markdown, speedups
from bench.runner import run
from bench.timing import Settings
from udfx.types import STR
from workloads.models import Workload


@pytest.mark.parametrize("status", ["runtime", "guarded"])
def test_runtime_reports_keep_execution_and_speedup_groups(status: str, tmp_path: Path) -> None:
    base = {
        "suite": "test",
        "engine": "spark",
        "mode": "query",
        "primary_baseline": "python",
        "native_row_fraction": 1.0,
    }
    results = [
        {
            **base,
            "workload": name,
            "status": state,
            "runtime_udfs": runtime,
            "comparisons": {"python": {"median_speedup": ratio}},
        }
        for name, state, runtime, ratio in (
            ("expression", "native", [], 1.0),
            ("runtime", status, ["spark.json_words"], 4.0),
            ("fallback", "python", [], 9.0),
        )
    ]
    artifact = {"commit": "test", "python": "3.12", "results": results}
    ratios, deduplicated = speedups(results)
    assert ratios == [1.0, 4.0]
    assert deduplicated == pytest.approx(2.0)
    report = markdown(artifact)
    assert "| 3.30x | 2.00x | 2.00x | 0/2 |" in report
    assert "| 1 | 1 | 0 | 1 |" in report if status == "runtime" else "| 1 | 0 | 1 | 1 |" in report
    assert f"| runtime | spark | query | {status} | spark.json_words |" in report
    assert "Non-Python rows" in report
    path = tmp_path / "results.csv"
    export_csv(artifact, path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["runtime_udfs"] == "spark.json_words"
    assert rows[1]["status"] == status


@pytest.mark.xdist_group("spark")
def test_benchmark_records_actual_spark_runtime(tmp_path: Path) -> None:
    def packed(text: str) -> str:
        return json.dumps(text.split())

    source = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.table({"text": ["one two", "", "a\tb"]}), source)
    workload = Workload("test", "runtime", packed, {"text": STR}, STR, source, {"text": "text"})
    results = run(workload, ["spark"], Settings(warmups=0, rounds=1), tmp_path / "inputs")
    assert len(results) == 2
    for result in results:
        assert result["status"] == "runtime", result.get("reason")
        assert result["runtime_udfs"] == ["spark.json_words"]
        assert result["native_row_fraction"] == 1.0
        assert result["compiled_udfs"] == 1
        assert result["comparisons"]
