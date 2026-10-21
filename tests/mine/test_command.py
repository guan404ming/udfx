import json
import sys
from pathlib import Path

import pytest

from bench.mine.command import main

from .helpers import collect_sources


def test_main_writes_report_and_candidate_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collect_sources(
        tmp_path,
        [("notebook", "offset = 2\nframe.map(lambda value: value + offset)")],
    )
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["bench.mine", str(tmp_path / "input.csv"), str(output)])
    main()
    records = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0] == {
        "kernel_id": "notebook",
        "kind": "element",
        "occurrences": 1,
        "source": "lambda value: value + offset",
        "ok": True,
        "schema": {"value": "float?"},
        "reason": None,
        "category": None,
        "third_party_names": [],
    }
    assert "Compiled unique: 1/1 (100.00%)" in (output / "summary.md").read_text()
