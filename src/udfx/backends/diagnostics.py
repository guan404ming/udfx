from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class LoweringDetails:
    exact: object | None
    value_guard: bool
    fault_guard: bool
    failure: bool
    native_errors: bool


CURRENT: ContextVar[list[LoweringDetails] | None] = ContextVar("udfx_diagnostics", default=None)


@contextmanager
def capture_lowering() -> Generator[list[LoweringDetails]]:
    details: list[LoweringDetails] = []
    token = CURRENT.set(details)
    try:
        yield details
    finally:
        CURRENT.reset(token)


def record_lowering(
    exact: object | None,
    *,
    value_guard: bool,
    fault_guard: bool,
    failure: bool,
    native_errors: bool,
) -> None:
    details = CURRENT.get()
    if details is not None:
        details[:] = [LoweringDetails(exact, value_guard, fault_guard, failure, native_errors)]
