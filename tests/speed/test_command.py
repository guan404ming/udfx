import importlib
from pathlib import Path

import pytest
from conftest import engine_names

from bench.speed import polars, spark
from bench.speed.command import main


@pytest.mark.parametrize(
    "engine",
    [pytest.param(name, id=name, marks=pytest.mark.xdist_group(name)) for name in engine_names()],
)
def test_engine_report(
    engine: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(f"bench.speed.{engine}")
    monkeypatch.setattr(module, "timed", lambda action, repeats: 1.0)
    main(
        [
            "--rows",
            "200",
            "--repeats",
            "1",
            "--engines",
            engine,
            "--cases",
            "lower_strip",
            "row_total",
            "type_identity",
            "instance_text",
        ]
    )
    output = capsys.readouterr().out
    assert output.startswith("rows 200, best of 1, load average ")
    assert "skipped" not in output, output
    tables = output.split("\n\n")
    details = [line.split(" | ") for line in tables[1].splitlines()[2:]]
    summaries = [line.split(" | ") for line in tables[2].splitlines()[2:]]
    engines = {"polars eager", "polars lazy"} if engine == "polars" else {engine}
    assert len(details) == 4 * len(engines)
    assert {row[0].removeprefix("| ") for row in details} == {
        "lower_strip",
        "row_total",
        "type_identity",
        "instance_text",
    }
    assert {row[1] for row in details} == engines
    for row in details:
        assert len(row) == 9
        assert row[2] in {"map_elements", "native", "arrow", "pickled"}
        assert float(row[3]) > 0
        assert float(row[4]) > 0
        assert row[5].endswith("x")
        assert row[6] in {"true", "false"}
        assert row[7].endswith("%")
        assert row[8].removesuffix(" |") in {"PASS", "FAIL"}
    assert "cases | groups | K2 fails | geomean | deduplicated geomean | K3" in tables[2]
    assert {row[0].removeprefix("| ") for row in summaries} == engines
    for row in summaries:
        assert len(row) == 8
        assert row[1] == "4"
        assert row[2] == "3"
        assert 0 <= int(row[3]) <= 4
        assert row[4].endswith("x")
        assert row[5].endswith("x")
        assert row[6] in {"PASS", "FAIL"}
        assert row[7].removesuffix(" |") in {"0/4", "1/4", "2/4", "3/4", "4/4"}
    duplicates = [line.split(" | ") for line in tables[3].splitlines()[2:]]
    assert len(duplicates) == len(engines)
    assert {row[0].removeprefix("| ") for row in duplicates} == engines
    for row in duplicates:
        assert len(row[1]) == 12
        assert row[2] == "type_identity, instance_text"
        assert len(row[3].removesuffix(" |").split(", ")) == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--rows", "0"],
        ["--repeats", "-1"],
        ["--engines", "unknown"],
        ["--cases", "unknown"],
    ],
)
def test_invalid_arguments(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("engine", ["polars", "spark"])
def test_mismatch_stops_before_timing(engine: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module(f"bench.speed.{engine}")
    monkeypatch.setattr(module, "caller", lambda program: lambda value: "wrong")
    monkeypatch.setattr(module, "timed", lambda *arguments: pytest.fail("timed incorrect output"))
    with pytest.raises(AssertionError, match=f"lower_strip on {engine}.* row 0"):
        main(["--rows", "200", "--engines", engine, "--cases", "lower_strip", "--repeats", "1"])


def test_baselines_are_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(polars, "timed", lambda action, repeats: calls.append(action) or 1.0)
    arguments = ["--rows", "200", "--repeats", "1", "--engines", "polars", "--cases", "lower_strip"]
    main([*arguments, "--baselines", str(tmp_path / "baselines.json")])
    first = len(calls)
    main([*arguments, "--baselines", str(tmp_path / "baselines.json")])
    assert first == 4
    assert len(calls) - first == 2


@pytest.mark.xdist_group("spark")
def test_spark_datetime_baseline_matches_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spark, "timed", lambda action, repeats: 1.0)
    main(["--rows", "200", "--repeats", "1", "--engines", "spark", "--cases", "datetime_format"])
