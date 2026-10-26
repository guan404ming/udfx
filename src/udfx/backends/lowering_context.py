from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from udfx.backends.rules import Lowered, Scope
from udfx.ir import Call, Guarded, If, Node
from udfx.semantics.assumptions import CONDITIONS, Conditions

if TYPE_CHECKING:
    from udfx.backends.base import Backend


@dataclass
class LoweringContext:
    binds: bool | None = None
    graph: bool = False
    has_bindings: bool = False
    results: dict[
        tuple[int, int, int, int, int],
        tuple[
            object, Node, Mapping[Node, Any], Conditions, Mapping[int, Node] | None, Lowered[Any]
        ],
    ] = field(default_factory=dict)


CURRENT: ContextVar[LoweringContext | None] = ContextVar("udfx_lowering", default=None)
EXACT: ContextVar[Mapping[int, Node] | None] = ContextVar("udfx_exact_inputs", default=None)


def evaluated_nodes(node: Node) -> dict[int, Node]:
    children = (
        node.arguments
        if isinstance(node, Call)
        else (node.condition,)
        if isinstance(node, If)
        else (node.value, node.exact)
        if isinstance(node, Guarded)
        else ()
    )
    found = {id(node): node}
    for child in children:
        found.update(evaluated_nodes(child))
    return found


@contextmanager
def assuming_exact(node: Node) -> Generator[None]:
    token = EXACT.set({**(EXACT.get() or {}), **evaluated_nodes(node)})
    try:
        yield
    finally:
        EXACT.reset(token)


@contextmanager
def lowering_context(
    *, binds: bool | None = None, graph: bool = False
) -> Generator[LoweringContext]:
    current = CURRENT.get()
    if current is not None and binds is None and not graph:
        yield current
        return
    context = LoweringContext(binds, graph)
    token = CURRENT.set(context)
    try:
        yield context
    finally:
        CURRENT.reset(token)


def bindings_enabled(default: bool) -> bool:
    context = CURRENT.get()
    return default if context is None or context.binds is None else context.binds


def graph_enabled() -> bool:
    context = CURRENT.get()
    return context is not None and context.graph


def lower_shared[E](backend: Backend[E], node: Node, elements: Scope[E]) -> Lowered[E]:
    with lowering_context() as context:
        conditions = CONDITIONS.get()
        exact = EXACT.get()
        key = id(backend), id(node), id(elements), id(conditions), id(exact)
        if key not in context.results:
            lowered = backend.lower_node(node, elements)
            if exact is not None and id(node) in exact:
                lowered = Lowered(lowered.expr, overflow=lowered.overflow)
            context.results[key] = (
                backend,
                node,
                elements,
                conditions,
                exact,
                lowered,
            )
        return context.results[key][5]
