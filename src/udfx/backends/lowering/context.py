from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from udfx.backends.rules import Lowered, Scope
from udfx.ir import Call, Guarded, If, Node
from udfx.semantics.numbers.assumptions import CONDITIONS, Conditions, known

if TYPE_CHECKING:
    from udfx.backends.base import Backend


@dataclass
class LoweringContext:
    binds: bool | None = None
    graph: bool = False
    has_bindings: bool = False
    runtime_udfs: set[str] = field(default_factory=set)
    results: dict[
        tuple[int, int, int, int, int],
        tuple[
            object, Node, Mapping[Node, Any], Conditions, Mapping[int, Node] | None, Lowered[Any]
        ],
    ] = field(default_factory=dict)
    guards: dict[int, tuple[object, tuple[object, ...]]] = field(default_factory=dict)
    keys: dict[int, tuple[object, str]] = field(default_factory=dict)


CURRENT: ContextVar[LoweringContext | None] = ContextVar("udfx_lowering", default=None)
EXACT: ContextVar[Mapping[int, Node] | None] = ContextVar("udfx_exact_inputs", default=None)


def evaluated_nodes(node: Node) -> dict[int, Node]:
    found: dict[int, Node] = {}
    pending = [node]
    conditions = dict(CONDITIONS.get())
    while pending:
        current = pending.pop()
        if id(current) in found:
            continue
        found[id(current)] = current
        if isinstance(current, Call):
            pending.extend(current.arguments)
        elif isinstance(current, If):
            pending.append(current.condition)
            selected = known(current.condition, conditions)
            if selected is not None:
                pending.append(current.then if selected else current.orelse)
        elif isinstance(current, Guarded):
            pending.extend((current.value, current.exact))
    return found


@contextmanager
def assuming_exact(node: Node) -> Generator[None]:
    token = EXACT.set({**(EXACT.get() or {}), **evaluated_nodes(node)})
    try:
        yield
    finally:
        EXACT.reset(token)


def guard_parts(guard: object) -> tuple[Any, ...]:
    context = CURRENT.get()
    found = None if context is None else context.guards.get(id(guard))
    return (guard,) if found is None else found[1]


def guard_key(guard: object) -> str:
    context = CURRENT.get()
    if context is None:
        return str(guard)
    found = context.keys.get(id(guard))
    if found is None:
        found = context.keys[id(guard)] = (guard, str(guard))
    return found[1]


def distinct_guards(parts: Iterable[object]) -> tuple[Any, ...]:
    present = {
        guard_key(guard): guard
        for part in parts
        if part is not None and part is not True
        for guard in guard_parts(part)
    }
    return tuple(present.values())


def remember_guards[E](guard: E, parts: tuple[E, ...]) -> E:
    context = CURRENT.get()
    if context is not None and len(parts) > 1:
        context.guards[id(guard)] = (guard, parts)
    return guard


def factored_guards(first: object, second: object) -> tuple[tuple[Any, ...], ...]:
    if first is None or second is None or first is False or second is False:
        return (), (), ()
    right = {guard_key(part) for part in guard_parts(second)}
    shared = {guard_key(part): part for part in guard_parts(first) if guard_key(part) in right}
    first_rest, second_rest = (
        tuple(part for part in guard_parts(value) if guard_key(part) not in shared)
        for value in (first, second)
    )
    return tuple(shared.values()), first_rest, second_rest


def record_runtime(name: str) -> None:
    context = CURRENT.get()
    if context is not None:
        context.runtime_udfs.add(name)


def runtime_udfs() -> tuple[str, ...]:
    context = CURRENT.get()
    return () if context is None else tuple(sorted(context.runtime_udfs))


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
    if current is not None:
        current.runtime_udfs.update(context.runtime_udfs)


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
                lowered = lowered.map(lambda value: replace(value, exact=None))
            context.results[key] = (
                backend,
                node,
                elements,
                conditions,
                exact,
                lowered,
            )
        return context.results[key][5]
