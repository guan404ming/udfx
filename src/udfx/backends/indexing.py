from __future__ import annotations

from dataclasses import replace

from udfx.backends.hoisting import reads_input
from udfx.ir import FALSE, TRUE, Call, Comprehension, Lit, Node, children
from udfx.ops import apply
from udfx.types import INT, Kind

UNROLL_LIMIT = 64


def range_stop(node: Comprehension) -> Node | None:
    source = node.source
    if isinstance(source, Call) and source.op.name == "range" and len(source.arguments) == 1:
        return source.arguments[0]
    return None


def literal_range(node: Comprehension) -> Comprehension | None:
    indexed = indexed_source(node)
    if indexed is None:
        return None
    items, stop = indexed
    if not isinstance(stop, Lit) or stop.type != INT or not isinstance(stop.value, int):
        return None
    if stop.value > UNROLL_LIMIT:
        return None
    node = restrict(node, apply("__lt__", node.element, apply("len", items)))
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
