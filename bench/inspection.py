import re
from collections.abc import Callable, Sequence
from typing import Any

from udfx.backends.base import Backend
from udfx.backends.program_lowering import lower_program
from udfx.program import Program
from udfx.testing.outcome import same


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
