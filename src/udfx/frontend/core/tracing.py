from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from udfx.ir import Node

FOLDED: ContextVar[dict[int, tuple[Node, Node]] | None] = ContextVar("udfx_folded", default=None)


@contextmanager
def joining_helpers() -> Generator[None]:
    token = FOLDED.set({})
    try:
        yield
    finally:
        FOLDED.reset(token)
