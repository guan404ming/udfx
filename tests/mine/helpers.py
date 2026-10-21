import csv
from pathlib import Path

from bench.mine.models import Candidate, Counts
from bench.mine.notebooks import collect_candidates


def collect_sources(tmp_path: Path, rows: list[tuple[str, str]]) -> tuple[list[Candidate], Counts]:
    path = tmp_path / "input.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kernel_id", "source"])
        writer.writerows(rows)
    return collect_candidates(path)
