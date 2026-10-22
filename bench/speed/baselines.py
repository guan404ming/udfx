import inspect
import json
import sys
from collections.abc import Callable
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from bench.speed import data
from bench.speed.timing import Baseline
from tests.corpus.cases import Case

LIBRARIES = ("polars", "duckdb", "datafusion", "pyspark", "pyarrow")


def installed(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"


class Baselines:
    def __init__(self, path: Path, rows: int, repeats: int) -> None:
        self.path = path
        self.entries: dict[str, list[float | str]] = (
            json.loads(path.read_text()) if path.exists() else {}
        )
        libraries = " ".join(f"{name} {installed(name)}" for name in LIBRARIES)
        self.context = f"{sys.version} {libraries} {rows} {repeats} {inspect.getsource(data)}"

    def for_case(self, case: Case) -> Baseline:
        source = f"{self.context} {inspect.getsource(case.function)} {case.schema}"
        digest = sha256(source.encode()).hexdigest()[:16]

        def lookup(engine: str, measure: Callable[[], tuple[float, str]]) -> tuple[float, str]:
            key = f"{engine} {case.name} {digest}"
            if key not in self.entries:
                self.entries[key] = list(measure())
                self.path.write_text(json.dumps(self.entries, indent=1, sort_keys=True))
            seconds, form = self.entries[key]
            return float(seconds), str(form)

        return lookup
