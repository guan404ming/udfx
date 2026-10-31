from pathlib import Path

import pytest

from bench.artifacts import source_digest


@pytest.mark.parametrize("suffix", [".java", ".jar"])
def test_source_hash_includes_runtime_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "src"
    directory.mkdir()
    artifact = directory / ("runtime" + suffix)
    artifact.write_bytes(b"first")
    before = source_digest()
    artifact.write_bytes(b"second")
    assert source_digest() != before
