import hashlib
import json
from pathlib import Path

import pytest

from bench.command import main, parse_arguments
from workloads import qure, udfbench
from workloads.catalog import DEFAULT_SUITES, load
from workloads.prepare import main as prepare


def test_local_suites_require_explicit_selection() -> None:
    assert parse_arguments([])[1].suites == list(DEFAULT_SUITES)
    assert "qure" not in DEFAULT_SUITES
    assert parse_arguments(["--suites", "qure"])[1].suites == ["qure"]


@pytest.mark.parametrize("arguments", [[], ["--source", "qure.code=missing"]])
def test_missing_local_inputs_do_not_create_cache(tmp_path: Path, arguments: list[str]) -> None:
    root = tmp_path / "cache"
    with pytest.raises(SystemExit) as error:
        prepare(["--suites", "qure", "--root", str(root), *arguments])
    assert error.value.code == 2
    assert not root.exists()


def test_case_selection_precedes_upstream_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "udfbench/source"
    directory.mkdir(parents=True)
    entries = []
    for name, source in {
        "lowerize": "def lowerize(val):\n    return val.lower()\n",
        "clean": 'raise BaseException("unselected module ran")\n',
    }.items():
        (directory / f"{name}.py").write_text(source)
        entries.append(
            {"path": f"{name}.py", "sha256": hashlib.sha256(source.encode()).hexdigest()}
        )
    monkeypatch.setattr(udfbench, "read_manifest", lambda _: {"files": entries})
    monkeypatch.setattr(
        udfbench, "BINDINGS", {name: udfbench.BINDINGS[name] for name in ["lowerize", "clean"]}
    )
    workloads = load(["udfbench"], tmp_path, {"udfbench/lowerize"})
    assert len(workloads) == 1
    assert workloads[0].error is None
    assert workloads[0].function is not None
    assert workloads[0].function("HELLO") == "hello"


def test_failed_verification_never_loads_external_definitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("bench.command.metadata", lambda: {"instrumented": False})

    def reject_definition(*args: object) -> None:
        pytest.fail("A definition was loaded before source verification")

    monkeypatch.setattr("workloads.qure.load_function", reject_definition)
    output = tmp_path / "results.json"
    assert (
        main(
            [
                "--suites",
                "qure",
                "--cases",
                "qure/froid/q2",
                "--engines",
                "polars",
                "--root",
                str(tmp_path / "missing"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    results = json.loads(output.read_text())["results"]
    assert len(results) == 2
    assert all(result["status"] == "error" for result in results)


def test_prepare_cases_select_only_their_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selections = []
    monkeypatch.setattr("workloads.prepare.resolve_sources", lambda *arguments: ([], []))
    monkeypatch.setattr("workloads.prepare.save_tables", lambda *arguments, **keywords: None)
    monkeypatch.setattr(qure, "prepare", lambda paths, selected: selections.append(selected) or {})
    sources = []
    for name in ("code", "tpch"):
        (tmp_path / name).mkdir()
        sources += ["--source", f"qure.{name}={tmp_path / name}"]
    prepare(["--suites", "qure", "--root", str(tmp_path), "--cases", "qure/froid/q6", *sources])
    assert selections == [{"lineitem_partsupp"}]
    with pytest.raises(SystemExit) as error:
        prepare(["--suites", "udfbench", "--cases", "qure/froid/q6"])
    assert error.value.code == 2
