from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields, replace

from udfx.graphs import map_children, substitute
from udfx.ir import Call, Col, Comprehension, Element, If, Lit, Node, children

type Literal = Callable[[Call, int], bool]


def factor_branches(roots: Sequence[Node], literal: Literal) -> tuple[Node, ...]:
    seen: dict[int, Node] = {}

    def visit(node: Node) -> Node:
        if id(node) not in seen:
            mapped = map_children(node, visit)
            seen[id(node)] = (factor(mapped, literal) if isinstance(mapped, If) else None) or mapped
        return seen[id(node)]

    return tuple(visit(root) for root in roots)


def factor(node: If, literal: Literal) -> Node | None:
    holes: dict[tuple[int, int], tuple[Element, Node, Node]] = {}
    memo: dict[tuple[int, int], Node] = {}

    def visit(left: Node, right: Node) -> Node:
        if left is right:
            return left
        key = (id(left), id(right))
        if key not in memo:
            memo[key] = merge(left, right) or hole(left, right)
        return memo[key]

    def hole(left: Node, right: Node) -> Node:
        if left.type != right.type:
            raise TypeError
        key = (id(left), id(right))
        if key not in holes:
            holes[key] = Element(f"udfx branch {len(holes)}", left.type), left, right
        return holes[key][0]

    def merge(left: Node, right: Node) -> Node | None:
        if type(left) is not type(right) or isinstance(left, Lit | Col | Element | Comprehension):
            return None
        changes: dict[str, object] = {}
        for field in fields(left):
            first, second = getattr(left, field.name), getattr(right, field.name)
            if isinstance(first, Node):
                changes[field.name] = visit(first, second)
            elif isinstance(first, tuple) and first and isinstance(first[0], Node):
                if len(first) != len(second):
                    return None
                changes[field.name] = tuple(map(visit, first, second))
            elif first != second:
                return None
        merged = replace(left, **changes)
        if isinstance(merged, Call) and any(
            literal(merged, index) and argument in elements()
            for index, argument in enumerate(merged.arguments)
        ):
            return None
        return merged

    def elements() -> set[Node]:
        return {element for element, _, _ in holes.values()}

    try:
        template = visit(node.then, node.orelse)
    except TypeError:
        return None
    if not holes or template in elements() or operations(template) <= len(holes):
        return None
    return substitute(
        template,
        {
            id(element): If(node.condition, left, right, element.type)
            for element, left, right in holes.values()
        },
    )


def operations(node: Node) -> int:
    seen: set[int] = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if id(current) not in seen and isinstance(current, Call | If):
            seen.add(id(current))
            pending.extend(children(current))
    return len(seen)
