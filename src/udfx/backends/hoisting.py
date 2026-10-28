from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from udfx.backends.rules import Lowered, Scope
from udfx.errors import Unsupported
from udfx.ir import Call, Col, Comprehension, Element, If, Node, children
from udfx.types import Kind

if TYPE_CHECKING:
    from udfx.backends.base import Backend


def free(node: Node) -> frozenset[Element]:
    seen: dict[int, frozenset[Element]] = {}

    def visit(part: Node) -> frozenset[Element]:
        if id(part) not in seen:
            if isinstance(part, Element):
                result = frozenset({part})
            elif isinstance(part, Comprehension):
                inner = visit(part.value) | visit(part.condition) | visit(part.fault)
                result = visit(part.source) | (inner - {part.element})
            else:
                result = frozenset().union(*(visit(child) for child in children(part)))
            seen[id(part)] = result
        return seen[id(part)]

    return visit(node)


def reads_input(node: Node) -> bool:
    seen: dict[int, bool] = {}

    def visit(part: Node) -> bool:
        if id(part) not in seen:
            seen[id(part)] = isinstance(part, Col | Element) or any(
                visit(child) for child in children(part)
            )
        return seen[id(part)]

    return visit(node)


def size(node: Node) -> int:
    seen: dict[int, int] = {}

    def count(part: Node) -> int:
        identifier = id(part)
        if identifier not in seen:
            seen[identifier] = 1 + sum(count(child) for child in children(part))
        return seen[identifier]

    return count(node)


def invariant(node: Comprehension, elements: Mapping[Node, object]) -> Node | None:
    found: Counter[Node] = Counter()

    def visit(part: Node) -> None:
        if (
            isinstance(part, Call)
            and part.type.kind is Kind.LIST
            and free(part) <= free(node)
            and reads_input(part)
            and part not in elements
        ):
            found[part] += 1
            return
        for child in children(part):
            visit(child)

    for part in (node.value, node.condition, node.fault):
        visit(part)
    return max(found, key=lambda part: (found[part], size(part)), default=None)


def exact_free[E](backend: Backend[E], node: Node, elements: Scope[E]) -> bool:
    try:
        lowered = backend.lower(node, elements)
    except Unsupported:
        return False
    return lowered.exact is None and lowered.overflow is None


def repeated_condition(node: If, elements: Mapping[Node, object]) -> Node | None:
    try:
        hash(node)
    except TypeError:
        return None
    candidates: dict[Node, None] = {}

    def collect(part: Node) -> None:
        if part in elements:
            return
        if (
            isinstance(part, Call | If)
            and part.type.kind in {Kind.STR, Kind.INT, Kind.LIST}
            and free(part) <= free(node)
            and reads_input(part)
        ):
            candidates[part] = None
        for child in children(part):
            collect(child)

    collect(node.condition)
    counts: Counter[Node] = Counter()

    def count(part: Node) -> None:
        if part in elements:
            return
        if part in candidates:
            counts[part] += 1
        for child in children(part):
            count(child)

    count(node)
    repeated = [part for part in candidates if worth_binding(part, counts[part])]
    return max(
        repeated, key=lambda part: (size(part) * (counts[part] - 1), size(part)), default=None
    )


def hoist[E](
    backend: Backend[E],
    node: Node,
    elements: Scope[E],
    hoisted: Node,
    lower: Callable[[Node, Scope[E]], Lowered[E]] | None = None,
) -> Lowered[E]:
    definition = backend.lower(hoisted, elements).expr
    found: list[Lowered[E]] = []

    def inner(reference: E) -> Lowered[E]:
        found.append((lower or backend.lower)(node, {**elements, hoisted: reference}))
        return found[-1]

    expr = backend.bind(definition, lambda reference: inner(reference).expr)

    def bound(field: str) -> E | None:
        if getattr(found[0], field) is None:
            return None
        return backend.bind(
            definition, lambda reference: cast("E", getattr(inner(reference), field))
        )

    return Lowered(expr, bound("exact"), bound("overflow"))


def worth_binding(node: Node, occurrences: int) -> bool:
    def cost(part: Node) -> int:
        parts = children(part)
        return int(bool(parts)) + sum(cost(child) for child in parts)

    return cost(node) * (occurrences - 1) > 3
