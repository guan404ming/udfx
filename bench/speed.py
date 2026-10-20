import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from corpus.cases import CASES, Case

from udfx import compile
from udfx.backends.polars.dtypes import dtype
from udfx.types import Kind, Type

ROWS = 1_000_000
REPEATS = 3
SAMPLE_DATES = ["2024-05-17", "1999-12-31", "", "20x4-01-01", " 123-1"]
SAMPLE_WORDS = ["Hello World", "funder::class::id", "a::b", "", "  padded  "]


def column(name: str, type: Type, seed: random.Random) -> pl.Series:
    match type.kind:
        case Kind.STR:
            pool = SAMPLE_DATES if "date" in name else SAMPLE_WORDS
            values = [seed.choice(pool) for _ in range(ROWS)]
        case Kind.INT:
            values = [seed.randint(-1000, 1000) for _ in range(ROWS)]
        case Kind.FLOAT:
            values = [seed.uniform(-1000.0, 1000.0) for _ in range(ROWS)]
        case Kind.BOOL:
            values = [seed.random() < 0.5 for _ in range(ROWS)]
        case _:
            raise ValueError(f"no generator for {type}")
    if type.nullable:
        values = [None if seed.random() < 0.1 else value for value in values]
    return pl.Series(name, values, dtype=dtype(type))


def frame(schema: Mapping[str, Type]) -> pl.DataFrame:
    seed = random.Random(0)
    return pl.DataFrame([column(name, type, seed) for name, type in schema.items()])


def python_expr(
    function: Callable[..., object], schema: Mapping[str, Type], result: Type
) -> pl.Expr:
    names = list(schema)
    packed = pl.struct([pl.col(name) for name in names])
    return packed.map_elements(
        lambda row: function(*(row[name] for name in names)), return_dtype=dtype(result)
    )


def timed(run: Callable[[], object]) -> float:
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - start)
    return best


def measure(case: Case) -> tuple[float, float, float] | None:
    translation = compile(case.function, case.schema)
    if translation.program is None or translation.expr is None:
        return None
    data = frame(case.schema)
    python = python_expr(case.function, case.schema, translation.program.result)
    native = translation.expr
    return (
        timed(lambda: data.select(python.alias("out"))),
        timed(lambda: data.select(native.alias("out"))),
        timed(lambda: data.lazy().select(native.alias("out")).collect()),
    )


def main(names: Sequence[str]) -> None:
    chosen = [
        case for case in CASES if case.expect == "native" and (not names or case.name in names)
    ]
    print(f"{'case':<24}{'python s':>10}{'eager s':>10}{'lazy s':>10}{'eager x':>9}{'lazy x':>9}")
    for case in chosen:
        result = measure(case)
        if result is None:
            continue
        python, eager, lazy = result
        row = f"{case.name:<24}{python:>10.3f}{eager:>10.3f}{lazy:>10.3f}"
        print(f"{row}{python / eager:>8.1f}x{python / lazy:>8.1f}x")


if __name__ == "__main__":
    main(sys.argv[1:])
