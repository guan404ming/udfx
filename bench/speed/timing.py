import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from udfx.backends.base import Backend
from udfx.ir import FALSE
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

    @property
    def speedup(self) -> float:
        return self.python / self.native


def timed(action: Callable[[], object], repeats: int) -> float:
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        best = min(best, time.perf_counter() - start)
    return best


def fallback_in_plan(backend: Backend[Any], program: Program) -> bool:
    return backend.lower(program.value).exact is not None or program.fault != FALSE


def verify(engine: str, case: str, actual: Sequence[object], expected: Sequence[object]) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"{case} on {engine}: {len(actual)} rows != {len(expected)} rows")
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        if not same(left, right):
            raise AssertionError(f"{case} on {engine} row {index}: {left!r} != {right!r}")
