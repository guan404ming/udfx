import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow
import pytest

from workloads import catalog, udfbench
from workloads.storage import FIXTURES, digest, download, save_tables, verify_tables


@pytest.mark.parametrize("suite", ["flights", "nyc311"])
def test_offline_fixture_provenance(suite: str) -> None:
    metadata = verify_tables(suite, FIXTURES)
    assert metadata["source"]["files"]
    assert all(record["rows"] > 0 for record in metadata["tables"].values())


def test_download_checks_cached_and_fresh_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"public data")
    checksum = digest(source)
    target = tmp_path / "destination"
    assert download(source.as_uri(), checksum, target).read_bytes() == b"public data"
    source.unlink()
    assert download(source.as_uri(), checksum, target) == target
    target.write_text("tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(source.as_uri(), checksum, target)


def test_invalid_download_never_becomes_a_cached_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("changed")
    target = tmp_path / "destination"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(source.as_uri(), "0" * 64, target)
    assert not target.exists()
    assert not target.with_suffix(".partial").exists()


def test_prepared_data_and_manifest_tampering_fail(tmp_path: Path) -> None:
    save_tables("flights", {"flights": pyarrow.table({"value": [1]})}, tmp_path)
    verify_tables("flights", tmp_path)
    path = tmp_path / "flights/manifest.json"
    metadata = json.loads(path.read_text())
    metadata["source"]["selection"] = "changed"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Source manifest changed"):
        verify_tables("flights", tmp_path)
    metadata["schema_version"] = 2
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unsupported prepared-data schema"):
        verify_tables("flights", tmp_path)
    save_tables("flights", {"flights": pyarrow.table({"value": [1]})}, tmp_path)
    (tmp_path / "flights/flights.parquet").write_text("changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_tables("flights", tmp_path)


def test_catalog_rejects_duplicate_identifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    workload = catalog.load(["flights"], FIXTURES)[0]
    monkeypatch.setitem(catalog.SUITES, "duplicate", lambda root: [workload, replace(workload)])
    with pytest.raises(ValueError, match="unique"):
        catalog.load(["duplicate"], FIXTURES)


def test_missing_upstream_source_is_an_error_not_unsupported(tmp_path: Path) -> None:
    workloads = udfbench.load(tmp_path)
    assert len(workloads) == 25
    assert all(workload.error for workload in workloads)
    assert sum(not workload.deterministic for workload in workloads) == 1


def test_upstream_loader_preserves_source_and_checks_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = "def lowerize(val):\n    return val.lower() if val else None\n"
    path = tmp_path / "udfbench/source/lowerize.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    entry = {"path": "lowerize.py", "sha256": hashlib.sha256(source.encode()).hexdigest()}
    monkeypatch.setattr(udfbench, "read_manifest", lambda _: {"files": [entry]})
    binding = udfbench.BINDINGS["lowerize"]
    monkeypatch.setattr(udfbench, "BINDINGS", {"lowerize": binding})
    workload = udfbench.load(tmp_path)[0]
    assert workload.function is not None and workload.function("TEXT") == "text"
    assert workload.error is None
    assert path.read_text() == source
    monkeypatch.setattr(
        udfbench, "BINDINGS", {"lowerize": (binding[0], binding[1], {}, binding[3])}
    )
    assert "Schema does not match" in str(udfbench.load(tmp_path)[0].error)
    path.write_text("changed")
    assert "Checksum mismatch" in str(udfbench.load(tmp_path)[0].error)
    monkeypatch.setattr(udfbench, "BINDINGS", {})
    with pytest.raises(ValueError, match="Every upstream scalar"):
        udfbench.load(tmp_path)
