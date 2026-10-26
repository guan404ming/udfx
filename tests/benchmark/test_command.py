import fcntl
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bench.command import main
from bench.runner import run
from bench.timing import Settings
from workloads.flights import load
from workloads.storage import FIXTURES

pytestmark = pytest.mark.xdist_group("benchmark-command")


@pytest.fixture
def metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "bench.command.metadata", lambda: {"schema_version": 1, "instrumented": False}
    )


@pytest.mark.usefixtures("metadata")
def test_command_writes_all_requested_results(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    assert (
        main(
            [
                "--suites",
                "flights",
                "--engines",
                "polars",
                "--root",
                str(FIXTURES),
                "--output",
                str(path),
                "--cases",
                "flights/year_month",
                "--rounds",
                "1",
                "--warmups",
                "0",
            ]
        )
        == 0
    )
    result = json.loads(path.read_text())
    assert result["complete"]
    assert len(result["results"]) == 2
    assert all(row["status"] == "native" for row in result["results"])
    assert all("native" in row["samples"][0] for row in result["results"])


@pytest.mark.usefixtures("metadata")
def test_missing_data_is_recorded_for_every_requested_case(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "--suites",
                "flights",
                "--engines",
                "polars",
                "duckdb",
                "--root",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    result = json.loads(output.read_text())
    assert len(result["results"]) == 16
    assert all(row["status"] == "error" for row in result["results"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--rows", "0"],
        ["--rounds", "0"],
        ["--warmups", "-1"],
        ["--threads", "0"],
        ["--engines", "missing"],
        ["--engines", "polars", "polars"],
        ["--suites", "flights", "flights"],
        ["--suites", "missing"],
        ["--cases", "missing"],
    ],
)
def test_invalid_arguments(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_instrumented_benchmarks_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bench.command.metadata", lambda: {"instrumented": True})
    with pytest.raises(SystemExit) as error:
        main(["--suites", "flights"])
    assert error.value.code == 2


@pytest.mark.usefixtures("metadata")
def test_concurrent_benchmark_is_rejected(tmp_path: Path) -> None:
    lock = Path(".cache/benchmark.lock")
    lock.parent.mkdir(exist_ok=True)
    with lock.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit) as error:
            main(
                [
                    "--suites",
                    "flights",
                    "--root",
                    str(FIXTURES),
                    "--output",
                    str(tmp_path / "result.json"),
                ]
            )
    assert error.value.code == 2


def test_runner_distinguishes_exclusions_unsupported_and_errors(tmp_path: Path) -> None:
    workload = load(FIXTURES)[0]
    settings = Settings(warmups=0, rounds=1)
    variants = [
        (replace(workload, exclusion="random"), "excluded"),
        (replace(workload, error="missing dependency", exclusion="random"), "error"),
        (replace(workload, function=None), "error"),
        (
            replace(workload, function=lambda delay, cancelled: json.loads(str(delay))),
            "unsupported",
        ),
        (replace(workload, table=tmp_path / "missing.parquet"), "error"),
    ]
    for case, status in variants:
        results = run(case, ["polars", "spark"], settings, tmp_path)
        assert len(results) == 4
        assert all(result["status"] == status for result in results)


@pytest.mark.usefixtures("metadata")
def test_interrupted_run_keeps_the_entire_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(*args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("bench.command.run", interrupt)
    output = tmp_path / "result.json"
    with pytest.raises(KeyboardInterrupt):
        main(
            [
                "--suites",
                "flights",
                "--engines",
                "polars",
                "--root",
                str(FIXTURES),
                "--output",
                str(output),
            ]
        )
    artifact = json.loads(output.read_text())
    assert not artifact["complete"]
    assert len(artifact["results"]) == 8
    assert all(result["status"] == "pending" for result in artifact["results"])
