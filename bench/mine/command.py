from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from bench.mine.measurement import measure
from bench.mine.models import Result
from bench.mine.notebooks import collect_candidates
from bench.mine.report import summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    started = perf_counter()
    candidates, counts = collect_candidates(arguments.csv)
    counts.collection_seconds = perf_counter() - started
    print(f"Read {counts.cells:,} cells; measure {len(candidates):,} unique candidates", flush=True)
    arguments.output.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    started = perf_counter()
    with (arguments.output / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for index, candidate in enumerate(candidates, 1):
            result = measure(candidate)
            results.append(result)
            stream.write(json.dumps(asdict(result), ensure_ascii=True) + "\n")
            if index % 1000 == 0:
                print(f"Measured {index:,}/{len(candidates):,}", flush=True)
    counts.compile_seconds = perf_counter() - started
    (arguments.output / "summary.md").write_text(summary(results, counts), encoding="utf-8")
    print(arguments.output / "summary.md")
