from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from udfx.testing.outcome import Outcome, Raised

CURRENT: ContextVar[set[str] | None] = ContextVar("udfx_test_dimensions", default=None)


@contextmanager
def capture() -> Generator[set[str]]:
    dimensions: set[str] = set()
    token = CURRENT.set(dimensions)
    try:
        yield dimensions
    finally:
        CURRENT.reset(token)


def record(row: Sequence[object], expected: Outcome | None) -> None:
    dimensions = CURRENT.get()
    if dimensions is not None:
        dimensions.add(
            "guarded"
            if expected is None
            else "faults"
            if isinstance(expected, Raised)
            else "values"
        )
        if any(value is None for value in row):
            dimensions.add("nulls")
