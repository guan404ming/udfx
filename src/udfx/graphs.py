from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from dataclasses import fields, replace

from udfx.ir import Call, Node, children


def deterministic(node: Node) -> bool:
    return (not isinstance(node, Call) or node.op.deterministic) and all(
        deterministic(part) for part in children(node)
    )


def map_children(node: Node, transform: Callable[[Node], Node]) -> Node:
    changes = {}
    for field in fields(node):
        value = getattr(node, field.name)
        if isinstance(value, Node):
            mapped = transform(value)
            if mapped is not value:
                changes[field.name] = mapped
        elif isinstance(value, tuple) and all(isinstance(part, Node) for part in value):
            mapped = tuple(transform(part) for part in value)
            if any(left is not right for left, right in zip(value, mapped, strict=True)):
                changes[field.name] = mapped
    return replace(node, **changes) if changes else node


def canonicalize(roots: Sequence[Node]) -> tuple[Node, ...]:
    seen: dict[int, Node] = {}
    canonical: dict[tuple[object, ...], Node] = {}

    def key(value: object) -> object:
        if isinstance(value, Node):
            return "node", id(value)
        if isinstance(value, tuple):
            return "tuple", tuple(key(part) for part in value)
        if isinstance(value, float):
            return float, pickle.dumps(value)
        try:
            hash(value)
        except TypeError:
            return type(value), pickle.dumps(value)
        return type(value), value

    def visit(node: Node) -> Node:
        if id(node) not in seen:
            mapped = map_children(node, visit)
            identity = (type(node), *(key(getattr(mapped, field.name)) for field in fields(node)))
            seen[id(node)] = canonical.setdefault(identity, mapped)
        return seen[id(node)]

    return tuple(visit(node) for node in roots)


def substitute(node: Node, replacements: dict[int, Node]) -> Node:
    seen: dict[int, Node] = {}

    def visit(part: Node) -> Node:
        if id(part) in replacements:
            return replacements[id(part)]
        if id(part) not in seen:
            seen[id(part)] = map_children(part, visit)
        return seen[id(part)]

    return visit(node)


def unique_size(roots: Sequence[Node]) -> int:
    seen: set[int] = set()
    pending = list(roots)
    while pending:
        node = pending.pop()
        if id(node) not in seen:
            seen.add(id(node))
            pending.extend(children(node))
    return len(seen)
