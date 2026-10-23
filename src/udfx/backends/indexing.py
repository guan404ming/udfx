from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from udfx.backends.hoisting import reads_input
from udfx.backends.registration import Lowered
from udfx.errors import Unsupported
from udfx.ir import FALSE, TRUE, Call, Comprehension, Lit, Node, children
from udfx.ops import OpKey, apply
from udfx.types import INT, Kind

if TYPE_CHECKING:
    from udfx.backends.base import Backend

UNROLL_LIMIT = 64


def range_stop(node: Comprehension) -> Node | None:
    source = node.source
    if isinstance(source, Call) and source.op.name == "range" and len(source.arguments) == 1:
        return source.arguments[0]
    return None


def literal_range(node: Comprehension) -> Comprehension | None:
    stop, indexed = range_stop(node), indexed_source(node)
    if indexed is not None:
        items, stop = indexed
        node = restrict(node, apply("__lt__", node.element, apply("len", items)))
    if not isinstance(stop, Lit) or stop.type != INT or not isinstance(stop.value, int):
        return None
    if stop.value > UNROLL_LIMIT:
        return None
    return replace(node, source=Lit(tuple(range(stop.value)), node.source.type))


def restrict(node: Comprehension, limit: Node) -> Comprehension:
    condition = limit if node.condition == TRUE else apply("and", limit, node.condition)
    fault = node.fault if node.fault == FALSE else apply("and", limit, node.fault)
    return replace(node, condition=condition, fault=fault)


def indexed_source(node: Comprehension) -> tuple[Node, Node | None] | None:
    stop = range_stop(node)
    if stop is None:
        return None
    bounds = stop.arguments if isinstance(stop, Call) and stop.op.name == "min" else (stop,)
    for bound in bounds:
        if isinstance(bound, Call) and bound.op.name == "len" and len(bound.arguments) == 1:
            items = bound.arguments[0]
            others = [other for other in bounds if other is not bound]
            constant = not any(reads_input(other) for other in others)
            if items.type.kind is Kind.LIST and len(others) <= 1 and constant:
                return items, others[0] if others else None
    return None


def item_lookups(node: Comprehension, items: Node) -> frozenset[Node]:
    found: set[Node] = set()

    def visit(part: Node) -> None:
        if (
            isinstance(part, Call)
            and part.op.name == "__getitem__"
            and part.arguments == (items, node.element)
        ):
            found.add(part)
            return
        for child in children(part):
            visit(child)

    for part in (node.value, node.condition, node.fault):
        visit(part)
    return frozenset(found)


def lower_indexed[E](
    backend: Backend[E],
    node: Comprehension,
    elements: Mapping[Node, E],
    items: Node,
    bound: Node | None,
) -> Lowered[E]:
    lookups = item_lookups(node, items)
    if bound is not None:
        node = restrict(node, apply("__lt__", node.element, bound))
    source = backend.lower(items, elements)
    guarded = False

    def parts(element: E, index: E) -> tuple[Lowered[E], Lowered[E], E | None]:
        nonlocal guarded
        scope = {**elements, node.element: index, **dict.fromkeys(lookups, element)}
        found = comprehension_parts(backend, node, scope)
        guarded = guarded or found[2] is not None
        return found

    def condition(element: E, index: E) -> E:
        return parts(element, index)[0].expr

    value = backend.list_indexed(
        source.expr,
        lambda element, index: parts(element, index)[1].expr,
        None if node.condition == TRUE else condition,
    )
    if not guarded and node.fault == FALSE:
        return Lowered(value, backend.exact(source))
    flags = backend.list_indexed(
        source.expr, lambda element, index: cast("E", parts(element, index)[2]), None
    )
    return Lowered(value, backend.conjunction(backend.exact(source), every(backend, flags)))


def comprehension_parts[E](
    backend: Backend[E], node: Comprehension, scope: Mapping[Node, E]
) -> tuple[Lowered[E], Lowered[E], E | None]:
    condition = backend.lower(node.condition, scope)
    value = backend.lower(node.value, scope)
    fault = backend.lower(node.fault, scope)
    value_exact = backend.exact(value)
    if node.condition != TRUE and value_exact is not None:
        value_exact = backend.select(condition.expr, value_exact, backend.literal(TRUE))
    exact = backend.conjunction(
        backend.exact(condition),
        value_exact,
        backend.exact(fault),
        backend.negate(fault.expr) if node.fault != FALSE else None,
    )
    return condition, value, exact


def every[E](backend: Backend[E], flags: E) -> E:
    reduce = backend.rules.get(OpKey("all", (Kind.LIST,)))
    if reduce is None:
        raise Unsupported(f"{backend.name} has no list reduction")
    return cast("E", reduce(flags))
