from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from functools import cache
from typing import TYPE_CHECKING, cast

from udfx.backends.rules import Lowered
from udfx.errors import Unsupported
from udfx.ir import Call, Col, Comprehension, Element, Node, children
from udfx.types import Kind

if TYPE_CHECKING:
    from udfx.backends.base import Backend


@cache
def free(node: Node) -> frozenset[Element]:
    if isinstance(node, Element):
        return frozenset({node})
    if isinstance(node, Comprehension):
        inner = free(node.value) | free(node.condition) | free(node.fault)
        return free(node.source) | (inner - {node.element})
    return frozenset().union(*(free(part) for part in children(node)))


@cache
def reads_input(node: Node) -> bool:
    return isinstance(node, Col | Element) or any(reads_input(part) for part in children(node))


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


def exact_free[E](backend: Backend[E], node: Node, elements: Mapping[Node, E]) -> bool:
    try:
        lowered = backend.lower(node, elements)
    except Unsupported:
        return False
    return lowered.exact is None and lowered.overflow is None


def hoist[E](
    backend: Backend[E], node: Node, elements: Mapping[Node, E], hoisted: Node
) -> Lowered[E]:
    definition = backend.lower(hoisted, elements).expr
    found: list[Lowered[E]] = []

    def inner(reference: E, pick: bool) -> E:
        found.append(backend.lower(node, {**elements, hoisted: reference}))
        return cast("E", backend.exact(found[-1]) if pick else found[-1].expr)

    expr = backend.bind(definition, lambda reference: inner(reference, False))
    if backend.exact(found[0]) is None:
        return Lowered(expr)
    return Lowered(expr, backend.bind(definition, lambda reference: inner(reference, True)))
