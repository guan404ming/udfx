from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from udfx.ir import Node

FOLDED: ContextVar[dict[int, tuple[Node, Node]] | None] = ContextVar("udfx_folded", default=None)
EFFECTS: ContextVar[set[str] | None] = ContextVar("udfx_effects", default=None)


@contextmanager
def collect_effects() -> Generator[set[str]]:
    effects: set[str] = set()
    token = EFFECTS.set(effects)
    try:
        yield effects
    finally:
        EFFECTS.reset(token)


@contextmanager
def joining_helpers() -> Generator[None]:
    token = FOLDED.set({})
    try:
        yield
    finally:
        FOLDED.reset(token)
