import argparse
import importlib
import sys
from collections.abc import Sequence

from bench.speed.data import dataset
from bench.speed.report import report
from bench.speed.timing import Measurement
from tests.corpus.cases import CASES
from udfx.frontend import trace

ENGINES = ("polars", "duckdb", "datafusion", "spark")


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(arguments: Sequence[str] | None = None) -> None:
    cases = {case.name: case for case in CASES if case.expect == "native"}
    parser = argparse.ArgumentParser(description="Compare udfx with Python UDFs on each engine.")
    parser.add_argument("--rows", type=positive_integer, default=1_000_000)
    parser.add_argument("--repeats", type=positive_integer, default=3)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--cases", nargs="+", choices=cases, default=list(cases))
    options = parser.parse_args(arguments)
    measurements: list[Measurement] = []
    skipped: dict[str, str] = {}
    engines = {}
    for engine in dict.fromkeys(options.engines):
        try:
            engines[engine] = importlib.import_module(f"bench.speed.{engine}")
        except ImportError as error:
            skipped[engine] = str(error)
    for name in dict.fromkeys(options.cases):
        case = cases[name]
        program = trace(case.function, case.schema)
        table, expected = dataset(case, program, options.rows)
        for engine, module in engines.items():
            try:
                measurements.extend(module.measure(name, program, table, expected, options.repeats))
            except AssertionError:
                raise
            except Exception as error:
                skipped[f"{name} / {engine}"] = f"{type(error).__name__}: {error}"
        print(f"done {name}", file=sys.stderr, flush=True)
    print(report(measurements, skipped, options.rows, options.repeats))
