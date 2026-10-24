from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from udfx.backends.rules import Lowered
from udfx.ir import Node
from udfx.semantics.assumptions import CONDITIONS, Conditions

if TYPE_CHECKING:
    from udfx.backends.base import Backend


@dataclass
class LoweringContext:
    binds: bool | None = None
    has_bindings: bool = False
    results: dict[
        tuple[int, int, int, int], tuple[object, Node, Mapping[Node, Any], Conditions, Lowered[Any]]
    ] = field(default_factory=dict)


CURRENT: ContextVar[LoweringContext | None] = ContextVar("udfx_lowering", default=None)


@contextmanager
def lowering_context(*, binds: bool | None = None) -> Generator[LoweringContext]:
    current = CURRENT.get()
    if current is not None and binds is None:
        yield current
        return
    context = LoweringContext(binds)
    token = CURRENT.set(context)
    try:
        yield context
    finally:
        CURRENT.reset(token)


def bindings_enabled(default: bool) -> bool:
    context = CURRENT.get()
    return default if context is None or context.binds is None else context.binds


def lower_shared[E](backend: Backend[E], node: Node, elements: Mapping[Node, E]) -> Lowered[E]:
    with lowering_context() as context:
        conditions = CONDITIONS.get()
        key = id(backend), id(node), id(elements), id(conditions)
        if key not in context.results:
            context.results[key] = (
                backend,
                node,
                elements,
                conditions,
                backend.lower_node(node, elements),
            )
        return context.results[key][4]
