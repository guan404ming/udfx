import json
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

import pytest

from bench.artifacts import metadata, read, source_digest, write
from bench.coverage import check_baseline, code_summary, semantic_summary
from bench.coverage import main as coverage_main
from bench.report import export_csv, main, markdown, speedups
from bench.timing import summarize
from udfx.ops import all_ops
from udfx.testing.observation import capture, record
from udfx.testing.outcome import Raised, Returned


def artifact() -> dict:
    comparison = summarize([{"python_scalar": 2.0, "udfx": 4.0}], "python_scalar")
    base = {"suite": "test", "engine": "polars", "mode": "memory", "workload": "test/case"}
    return {
        "schema_version": 1,
        "commit": "test",
        "python": "3.12 test",
        "results": [
            {
                **base,
                "status": "native",
                "primary_baseline": "python_scalar",
                "native_row_fraction": 1.0,
                "comparisons": {"python_scalar": comparison},
            },
            {**base, "status": "unsupported", "reason": "unsupported function"},
            {**base, "status": "error", "reason": "wrong output"},
            {**base, "status": "excluded", "reason": "random"},
        ],
    }


def test_artifact_round_trip_and_future_schema(tmp_path: Path) -> None:
    path = tmp_path / "nested/result.json"
    write(path, artifact())
    assert read(path) == artifact()
    write(path, {"schema_version": 99})
    with pytest.raises(ValueError, match="Unsupported result schema"):
        read(path)


def test_metadata_identifies_code_and_machine() -> None:
    result = metadata()
    assert len(result["commit"]) == 40
    assert len(result["source_sha256"]) == 64
    assert result["logical_cpus"] > 0
    assert result["processor"]
    assert set(result["versions"]) == {
        "udfx",
        "pyarrow",
        "polars",
        "duckdb",
        "datafusion",
        "pyspark",
    }


def test_source_hash_includes_uncommitted_workload_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "workloads"
    directory.mkdir()
    path = directory / "case.py"
    path.write_text("value = 1\n")
    first = source_digest()
    path.write_text("value = 2\n")
    assert source_digest() != first
    (directory / "ignored.txt").write_text("unchanged code")
    second = source_digest()
    (directory / "ignored.txt").write_text("new text")
    assert source_digest() == second


def test_reports_include_failures_and_slower_cases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = artifact()
    text = markdown(result)
    assert "| 1 | 0 | 1 | 1 | 1 | 0 | 0.50x | 0.50x | 1/1 |" in text
    assert "wrong output" in text
    path = tmp_path / "result.csv"
    export_csv(result, path)
    assert len(path.read_text().splitlines()) == 5
    saved = tmp_path / "result.json"
    write(saved, result)
    main([str(saved)])
    assert capsys.readouterr().out == text
    report = tmp_path / "report.md"
    main([str(saved), "--markdown", str(report), "--csv", str(path)])
    assert report.read_text() == text
    result["results"] = result["results"][1:]
    assert "N/A" in markdown(result)


def test_duplicate_plans_have_one_group_weight() -> None:
    results = [
        {
            "workload": name,
            "status": "native",
            "primary_baseline": "python",
            "fingerprint": fingerprint,
            "comparisons": {"python": {"median_speedup": ratio}},
        }
        for name, fingerprint, ratio in [
            ("one", "first", 1.0),
            ("alias", "first", 1.0),
            ("two", "second", 9.0),
        ]
    ]
    ratios, dedup = speedups(results)
    assert ratios == [1.0, 1.0, 9.0]
    assert dedup == pytest.approx(3.0)


def test_code_summary_keeps_framework_and_compiler_separate() -> None:
    summary = {"covered_lines": 9, "num_statements": 10, "covered_branches": 3, "num_branches": 4}
    report = {
        "files": {
            name: {"summary": summary}
            for name in [
                "src/udfx/api.py",
                "bench/runner.py",
                "workloads/flights.py",
                "tests/test_api.py",
                "workloads/corpus/cases.py",
                "bench/mine/command.py",
            ]
        }
    }
    result = code_summary(report)
    assert result["compiler"]["line_percent"] == 90
    assert result["framework"]["covered_lines"] == 18
    assert result["framework"]["branch_percent"] == 75
    assert check_baseline(result, {"compiler": {"line_percent": 91}})
    assert check_baseline(result, {"compiler": {"line_percent": 90}}) == []
    empty = code_summary({"files": {}})
    assert check_baseline(empty, {"compiler": {"line_percent": 90}})


def test_semantic_matrix_uses_executed_results_and_keeps_missing(tmp_path: Path) -> None:
    root = Element("testsuite")
    signature = str(all_ops()[0])
    for outcome in ("passed", "failure", "skipped"):
        case = SubElement(root, "testcase", name=outcome)
        properties = SubElement(case, "properties")
        for name, value in {
            "signature": signature,
            "backend": "polars",
            "dimensions": "values,nulls",
        }.items():
            SubElement(properties, "property", name=name, value=value)
        if outcome != "passed":
            SubElement(case, outcome)
    SubElement(root, "testcase", name="not_conformance")
    path = tmp_path / "results.xml"
    ElementTree(root).write(path)
    summary = semantic_summary([path])
    assert summary["counts"]["failed"] == 1
    assert summary["counts"]["not_run"] == len(all_ops()) * 4 - 1
    row = next(row for row in summary["matrix"] if row["status"] == "failed")
    assert row["dimensions"] == ["nulls", "values"]


def test_coverage_command_reports_missing_signatures_and_floor_failures(tmp_path: Path) -> None:
    code = tmp_path / "code.json"
    code.write_text(json.dumps({"files": {}}))
    junit = tmp_path / "tests.xml"
    junit.write_text("<testsuite/>")
    output = tmp_path / "coverage.json"
    arguments = ["--code", str(code), "--junit", str(junit), "--output", str(output)]
    assert coverage_main(arguments) == 1
    assert json.loads(output.read_text())["semantic"]["counts"]["not_run"] > 0
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"compiler": {"line_percent": 90}}))
    assert coverage_main([*arguments, "--baseline", str(baseline)]) == 1


def test_semantic_observation_does_not_run_the_oracle_twice() -> None:
    record([1], Returned(1))
    with capture() as outer:
        record([1], Returned(1))
        with capture() as inner:
            record([None], Raised(TypeError))
            record([2], None)
        assert inner == {"nulls", "faults", "guarded"}
        assert outer == {"values"}
    record([None], None)
    assert outer == {"values"}
