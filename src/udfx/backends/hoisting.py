from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from functools import cache

from udfx.ir import Call, Col, Comprehension, Element, Node, children
from udfx.types import Kind


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


@cache
def size(node: Node) -> int:
    return 1 + sum(size(part) for part in children(node))


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
    best = max(found, key=lambda part: (found[part], size(part)), default=None)
    return best if best is not None and found[best] > 1 else None
