import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from udfx.backends.base import Backend
from udfx.backends.lowering import lower_program
from udfx.program import Program
from udfx.testing.outcome import same


@dataclass(frozen=True)
class Measurement:
    case: str
    engine: str
    python: float
    native: float
    form: str
    fallback: bool
    fingerprint: str
    pending: float

    @property
    def speedup(self) -> float:
        return self.python / self.native


Baseline = Callable[[str, Callable[[], tuple[float, str]]], tuple[float, str]]


def measured(engine: str, measure: Callable[[], tuple[float, str]]) -> tuple[float, str]:
    return measure()


def timed(action: Callable[[], object], repeats: int) -> float:
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        best = min(best, time.perf_counter() - start)
    return best


def fallback_in_plan(expr: Any, engine: str) -> bool:
    if engine == "polars":
        return '"AnonymousFunction":' in expr.meta.serialize(format="json")
    text = str(expr)
    if engine == "duckdb":
        text = re.sub(r"'(?:''|[^'])*'", "", text)
    if engine == "datafusion":
        text = re.sub(r'Utf8\("(?:\\.|[^"\\])*"\)', "", text)
    return bool(re.search(r"udfx fallback [^()]+\(|\bpatch\(", text))


def pending_share(
    backend: Backend[Any], program: Program, evaluate: Callable[[Any], Sequence[object]]
) -> float:
    exact = lower_program(backend, program).exact
    if exact is None:
        return 0.0
    flags = evaluate(exact)
    return sum(flag is not True for flag in flags) / len(flags) if flags else 0.0


def verify(engine: str, case: str, actual: Sequence[object], expected: Sequence[object]) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"{case} on {engine}: {len(actual)} rows != {len(expected)} rows")
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        if not same(left, right):
            raise AssertionError(f"{case} on {engine} row {index}: {left!r} != {right!r}")
