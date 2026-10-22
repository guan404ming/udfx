import pytest

from bench.speed import polars
from bench.speed.command import main
from udfx.backends import installed


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(backend.name, id=backend.name, marks=pytest.mark.xdist_group(backend.name))
        for backend in installed()
    ],
)
def test_engine_report(engine: str, capsys: pytest.CaptureFixture[str]) -> None:
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
        ]
    )
    output = capsys.readouterr().out
    assert output.startswith("rows 200, best of 1, load average ")
    assert "skipped" not in output, output
    tables = output.split("\n\n")
    details = [line.split(" | ") for line in tables[1].splitlines()[2:]]
    summaries = [line.split(" | ") for line in tables[2].splitlines()[2:]]
    engines = {"polars eager", "polars lazy"} if engine == "polars" else {engine}
    assert len(details) == 2 * len(engines)
    assert {row[0].removeprefix("| ") for row in details} == {"lower_strip", "row_total"}
    assert {row[1] for row in details} == engines
    for row in details:
        assert len(row) == 8
        assert row[2] in {"map_elements", "native", "arrow", "pickled"}
        assert float(row[3]) > 0
        assert float(row[4]) > 0
        assert row[5].endswith("x")
        assert row[6] in {"true", "false"}
        assert row[7].removesuffix(" |") in {"PASS", "FAIL"}
    assert "K2 fails | geomean | K3 | K5 no fallback" in tables[2]
    assert {row[0].removeprefix("| ") for row in summaries} == engines
    for row in summaries:
        assert len(row) == 6
        assert row[1] == "2"
        assert 0 <= int(row[2]) <= 2
        assert row[4] in {"PASS", "FAIL"}
        assert row[5].removesuffix(" |") in {"0/2", "1/2", "2/2"}


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


def test_mismatch_stops_before_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(polars, "caller", lambda program: lambda value: "wrong")
    monkeypatch.setattr(polars, "timed", lambda *arguments: pytest.fail("timed incorrect output"))
    with pytest.raises(AssertionError, match="lower_strip on polars eager row 0"):
        main(["--rows", "200", "--engines", "polars", "--cases", "lower_strip", "--repeats", "1"])
