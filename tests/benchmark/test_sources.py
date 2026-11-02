import json
from pathlib import Path

import pyarrow
import pytest

from workloads import source, storage
from workloads.source import parse_sources, resolve_sources
from workloads.storage import digest, save_tables, verify_tables


def test_local_sources_keep_hashes_without_private_paths(tmp_path: Path) -> None:
    origin = tmp_path / "private"
    origin.mkdir()
    (origin / "function.py").write_text("def value(number):\n    return number + 1\n")
    (origin / "data.tbl").write_text("1|value|\n")
    entries = [
        {"path": "function.py", "input": "suite.code", "sha256": digest(origin / "function.py")},
        {"path": "data.tbl", "input": "suite.data"},
    ]
    destination = tmp_path / "cached"
    paths, records = resolve_sources(
        {"files": entries}, destination, {"suite.code": origin, "suite.data": origin}
    )
    assert paths == [destination / entry["path"] for entry in entries]
    assert records == [{**entry, "sha256": digest(origin / entry["path"])} for entry in entries]
    assert str(tmp_path) not in json.dumps(records)
    assert all(path.read_bytes() == (origin / path.name).read_bytes() for path in paths)
    assert resolve_sources(
        {"files": entries}, destination, {"suite.code": origin, "suite.data": origin}
    ) == (paths, records)


def test_url_sources_require_a_pin_and_keep_the_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = {"path": "data.txt", "url": "https://example.org/data", "sha256": "a" * 64}
    calls = []

    def download(url: str, checksum: str, destination: Path) -> Path:
        calls.append((url, checksum, destination))
        return destination

    monkeypatch.setattr(source, "download", download)
    paths, records = resolve_sources({"files": [entry]}, tmp_path, {})
    assert calls == [(entry["url"], entry["sha256"], tmp_path / "data.txt")]
    assert paths == [tmp_path / "data.txt"]
    assert records == [entry]
    del entry["sha256"]
    with pytest.raises(ValueError, match="needs a SHA-256"):
        resolve_sources({"files": [entry]}, tmp_path, {})


@pytest.mark.parametrize("change", ["root", "file", "source", "cache"])
def test_local_cache_never_replaces_current_source_validation(tmp_path: Path, change: str) -> None:
    origin = tmp_path / "original"
    origin.mkdir()
    path = origin / "data.txt"
    path.write_text("original")
    manifest = {"files": [{"path": path.name, "input": "data"}]}
    destination = tmp_path / "cached"
    sources = {"data": origin}
    resolve_sources(manifest, destination, sources)
    if change == "root":
        sources.clear()
    elif change == "file":
        path.unlink()
    elif change == "source":
        path.write_text("changed")
    else:
        (destination / path.name).write_text("changed")
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_sources(manifest, destination, sources)


def test_pinned_local_source_rejects_changed_original_with_valid_cache(tmp_path: Path) -> None:
    origin = tmp_path / "original"
    origin.mkdir()
    path = origin / "function.py"
    path.write_text("value = 1\n")
    manifest = {"files": [{"path": path.name, "input": "code", "sha256": digest(path)}]}
    resolve_sources(manifest, tmp_path / "cache", {"code": origin})
    path.write_text("value = 2\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        resolve_sources(manifest, tmp_path / "cache", {"code": origin})


@pytest.mark.parametrize("name", ["../outside", "inside/../outside", "/outside", "", ".", None])
def test_source_paths_reject_traversal(tmp_path: Path, name: object) -> None:
    with pytest.raises(ValueError, match="Source path"):
        resolve_sources({"files": [{"path": name, "input": "data"}]}, tmp_path, {})


@pytest.mark.parametrize("location", ["input", "cache"])
def test_source_paths_reject_symlink_escape(tmp_path: Path, location: str) -> None:
    origin = tmp_path / "original"
    destination = tmp_path / "cache"
    outside = tmp_path / "outside"
    for directory in (origin, destination, outside):
        directory.mkdir()
    (outside / "data.txt").write_text("outside")
    (origin if location == "input" else destination).joinpath("nested").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(ValueError, match="Source path must stay inside"):
        resolve_sources(
            {"files": [{"path": "nested/data.txt", "input": "data"}]},
            destination,
            {"data": origin},
        )


@pytest.mark.parametrize(
    "entries",
    [
        [{"path": "data.txt"}],
        [{"path": "data.txt", "url": "https://example.org/data", "input": "data"}],
        [{"path": "data.txt", "input": "data", "sha256": "invalid"}],
        [{"path": "data.txt", "url": "", "sha256": "a" * 64}],
        [{"path": "data.txt", "input": "/private/data"}],
        [{"path": "data.txt", "input": "data"}] * 2,
        ["data.txt"],
    ],
)
def test_source_manifest_rejects_invalid_records(tmp_path: Path, entries: list) -> None:
    with pytest.raises(ValueError):
        resolve_sources({"files": entries}, tmp_path, {})


def test_parse_sources_preserves_named_directories(tmp_path: Path) -> None:
    directory = tmp_path / "with spaces=code"
    directory.mkdir()
    assert parse_sources([f"suite.code={directory}", f"suite.data={tmp_path}"]) == {
        "suite.code": directory,
        "suite.data": tmp_path,
    }
    with pytest.raises(ValueError, match="Duplicate local source"):
        parse_sources([f"data={tmp_path}", f"data={directory}"])


@pytest.mark.parametrize("value", ["data", "=path", "data=", "bad name=path", "../data=path"])
def test_parse_sources_rejects_malformed_assignments(value: str) -> None:
    with pytest.raises(ValueError, match="Expected NAME=DIR"):
        parse_sources([value])


def test_parse_sources_requires_a_directory(tmp_path: Path) -> None:
    path = tmp_path / "file"
    for exists in (False, True):
        if exists:
            path.write_text("file")
        with pytest.raises(ValueError, match="not a directory"):
            parse_sources([f"data={path}"])


@pytest.mark.parametrize("change", ["cache", "inventory", "provenance", "pin", "missing"])
def test_prepared_tables_verify_resolved_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    path = tmp_path / "data.txt"
    path.write_text("input")
    manifest = {"files": [{"path": path.name, "input": "data", "sha256": digest(path)}]}
    monkeypatch.setattr(storage, "read_manifest", lambda _: manifest)
    paths, records = resolve_sources(manifest, tmp_path / "suite/source", {"data": tmp_path})
    save_tables("suite", {"values": pyarrow.table({"value": [1]})}, tmp_path, sources=records)
    metadata = verify_tables("suite", tmp_path)
    assert metadata["sources"] == records
    if change == "cache":
        paths[0].write_text("changed")
    elif change == "inventory":
        metadata["sources"] = []
    elif change == "provenance":
        metadata["sources"][0]["input"] = "other"
    elif change == "missing":
        del metadata["sources"]
    else:
        paths[0].write_text("changed")
        metadata["sources"][0]["sha256"] = digest(paths[0])
    (tmp_path / "suite/manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=r"Checksum mismatch|Resolved source"):
        verify_tables("suite", tmp_path)


def test_sources_reject_invalid_manifests_and_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a files list"):
        resolve_sources({}, tmp_path, {})
    root = tmp_path / "root"
    root.write_text("")
    manifest = {"files": [{"path": "data.txt", "input": "data"}]}
    with pytest.raises(ValueError, match="not a directory"):
        resolve_sources(manifest, tmp_path / "cached", {"data": root})
