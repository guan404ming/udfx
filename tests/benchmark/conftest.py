from pathlib import Path

import pytest

from tests.benchmark.samples import SUITE, register, table
from workloads.storage import save_tables


@pytest.fixture
def sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    register(monkeypatch)
    root = tmp_path / "data"
    save_tables(SUITE, {"rows": table()}, root)
    return root
