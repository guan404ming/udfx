from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from udfx.ir import Comprehension, Node

if TYPE_CHECKING:
    from udfx.backends.rules import Lowered

type Siblings = dict[int, tuple[Comprehension, ...]]
type View[E] = Callable[[E], Lowered[E]]

SHARED: ContextVar[dict[int, tuple[Lowered[Any], Any, View[Any]]] | None] = ContextVar(
    "udfx_shared", default=None
)

SIBLINGS: ContextVar[tuple[Siblings, dict[int, object]] | None] = ContextVar(
    "udfx_siblings", default=None
)


def siblings(nodes: Sequence[Node]) -> Siblings:
    found: dict[tuple[int, int], list[Comprehension]] = {}
    for node in nodes:
        if isinstance(node, Comprehension):
            found.setdefault((id(node.source), id(node.element)), []).append(node)
    return {
        id(member): tuple(members)
        for members in found.values()
        if len(members) > 1
        for member in members
    }


@contextmanager
def fusing(nodes: Sequence[Node]) -> Generator[None]:
    token = SIBLINGS.set((siblings(nodes), {}))
    try:
        yield
    finally:
        SIBLINGS.reset(token)


@contextmanager
def sharing() -> Generator[None]:
    token = SHARED.set({})
    try:
        yield
    finally:
        SHARED.reset(token)


def share[E](value: Lowered[E], base: E, view: View[E]) -> Lowered[E]:
    found = SHARED.get()
    if found is not None:
        found[id(value)] = value, base, view
    return value


def shared[E](value: Lowered[E]) -> tuple[E, View[E]] | None:
    found = (SHARED.get() or {}).get(id(value))
    if found is None or found[0] is not value:
        return None
    return found[1], found[2]
