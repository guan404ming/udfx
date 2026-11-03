import json
from pathlib import Path

import pyarrow
import pytest
from pyarrow import parquet

from workloads.prepare import PREPARERS, main, prepare, udfbench
from workloads.storage import digest, verify_tables


def test_udfbench_preparation_keeps_original_values(tmp_path: Path) -> None:
    path = tmp_path / "artifact_authorlists.parquet"
    parquet.write_table(pyarrow.table({"0": ["a", "b"], "1": ['["A"]', None]}), path)
    table = udfbench([path])["artifact_authorlists"]
    assert table["1"].to_pylist() == ['["A"]', None]
    assert table["next_authorlist"].to_pylist() == [None, '["A"]']


def test_explicit_prepare_checks_source_and_records_row_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact_authorlists.parquet"
    parquet.write_table(pyarrow.table({"1": ["a", "b"]}), path)
    entry = {"url": path.as_uri(), "sha256": digest(path), "path": path.name}
    source = {"files": [entry]}
    monkeypatch.setattr("workloads.prepare.read_manifest", lambda _: source)
    monkeypatch.setattr("workloads.storage.read_manifest", lambda _: source)
    root = tmp_path / "cache"
    prepare("udfbench", root, rows=1)
    metadata = verify_tables("udfbench", root)
    assert metadata["tables"]["artifact_authorlists"]["rows"] == 1
    assert metadata["sampling"] == {"method": "source prefix", "row_limit": 1}
    assert json.loads((root / "udfbench/manifest.json").read_text())["tables"]


def test_prepare_command_does_not_run_without_valid_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr("workloads.prepare.prepare", lambda *args: calls.append(args))
    with pytest.raises(SystemExit):
        main(["--rows", "0"])
    assert calls == []
    main(["--suites", "udfbench", "--root", str(tmp_path), "--rows", "3"])
    assert calls == [("udfbench", tmp_path, 3, {}, None)]


def test_sampling_keeps_the_complete_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {"files": [], "dimension_tables": ["dimension"]}
    monkeypatch.setattr("workloads.prepare.read_manifest", lambda _: source)
    monkeypatch.setattr("workloads.storage.read_manifest", lambda _: source)
    tables = {
        "facts": pyarrow.table({"value": [1, 2, 3]}),
        "dimension": pyarrow.table({"value": [4, 5, 6]}),
    }
    monkeypatch.setitem(PREPARERS, "udfbench", lambda _: tables)
    prepare("udfbench", tmp_path, rows=1)
    metadata = verify_tables("udfbench", tmp_path)
    assert metadata["tables"]["facts"]["rows"] == 1
    assert metadata["tables"]["dimension"]["rows"] == 3
