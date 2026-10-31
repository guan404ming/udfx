from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from udfx.backends.guarantees import required
from udfx.backends.hoisting import reads_input
from udfx.backends.registration import Lowered
from udfx.backends.rules import Scope
from udfx.errors import Unsupported
from udfx.ir import FALSE, TRUE, Call, Comprehension, Lit, Node, children
from udfx.ops import OpKey, apply
from udfx.semantics.assumptions import assuming
from udfx.semantics.facts import element_predicate
from udfx.types import INT, Kind

if TYPE_CHECKING:
    from udfx.backends.base import Backend

UNROLL_LIMIT = 64

type Parts[E] = tuple[Lowered[E], Lowered[E], E | None, E]


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


def matching(node: Comprehension, accept: Callable[[Node], bool]) -> frozenset[Node]:
    found: set[Node] = set()
    seen: set[int] = set()

    def visit(part: Node) -> None:
        if id(part) in seen:
            return
        seen.add(id(part))
        if accept(part):
            found.add(part)
            return
        for child in children(part):
            visit(child)

    for part in (node.value, node.condition, node.fault):
        visit(part)
    return frozenset(found)


def item_lookups(node: Comprehension, items: Node) -> frozenset[Node]:
    return matching(
        node,
        lambda part: (
            isinstance(part, Call)
            and part.op.name == "__getitem__"
            and part.arguments == (items, node.element)
        ),
    )


def index_offsets(node: Comprehension) -> frozenset[Node]:
    def offset(part: Node) -> bool:
        if not isinstance(part, Call) or part.op.name != "__add__" or len(part.arguments) != 2:
            return False
        left, right = part.arguments
        other = right if left == node.element else left if right == node.element else None
        return (
            isinstance(other, Lit)
            and other.type == INT
            and isinstance(other.value, int)
            and other.value <= 1
        )

    return matching(node, offset)


def lower_indexed[E](
    backend: Backend[E],
    node: Comprehension,
    elements: Scope[E],
    items: Node,
    bound: Node | None,
) -> Lowered[E]:
    lookups, offsets = item_lookups(node, items), index_offsets(node)
    if bound is not None:
        node = restrict(node, apply("__lt__", node.element, bound))
    source = backend.lower(items, elements)
    guarded = False

    def parts(element: E, index: E) -> Parts[E]:
        nonlocal guarded
        scope = {**elements, node.element: index, **dict.fromkeys(lookups, element)}
        for offset in offsets:
            lowered = backend.lower(offset, scope)
            if lowered.exact is None:
                scope = {**scope, offset: lowered.expr}
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
    return comprehension_result(
        backend,
        node,
        source,
        value,
        guarded=guarded,
        flags=lambda pick: backend.list_indexed(
            source.expr, lambda element, index: pick(parts(element, index)), None
        ),
    )


def comprehension_parts[E](backend: Backend[E], node: Comprehension, scope: Scope[E]) -> Parts[E]:
    with assuming(element_predicate(node.source, node.element), True):
        condition = backend.lower(node.condition, scope)
        with assuming(node.condition, True):
            value = backend.lower(node.value, scope)
        fault = backend.lower(node.fault, scope)
    value_exact = backend.exact(value)
    if node.condition != TRUE and value_exact is not None:
        value_exact = backend.select(condition.expr, value_exact, backend.literal(TRUE))
    exact = backend.conjunction(backend.exact(condition), value_exact, backend.exact(fault))
    return condition, value, exact, fault.expr


def comprehension_result[E](
    backend: Backend[E],
    node: Comprehension,
    source: Lowered[E],
    value: E,
    *,
    guarded: bool,
    flags: Callable[[Callable[[Parts[E]], E]], E],
) -> Lowered[E]:
    exact = backend.exact(source)
    if guarded:
        exact = backend.conjunction(exact, every(backend, flags(lambda found: cast("E", found[2]))))
    if node.fault == FALSE:
        return Lowered(value, exact, guarantees=required((source,)))
    safe = every(backend, flags(lambda found: backend.negate(found[3])))
    return Lowered(value, exact, backend.negate(safe), guarantees=required((source,)))


def every[E](backend: Backend[E], flags: E) -> E:
    reduce = backend.rules.get(OpKey("all", (Kind.LIST,)))
    if reduce is None:
        raise Unsupported(f"{backend.name} has no list reduction")
    return cast("E", reduce(flags))
