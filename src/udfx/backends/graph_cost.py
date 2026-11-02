from collections.abc import Mapping, Sequence

from udfx.backends.graph import layers, needed
from udfx.ir import Col, Element, Node, children


def operation_count(node: Node, references: Mapping[int, int] | None = None) -> int:
    costs = dict(references or {})

    def count(part: Node) -> int:
        identifier = id(part)
        if identifier not in costs:
            parts = children(part)
            costs[identifier] = int(bool(parts)) + sum(count(child) for child in parts)
        return costs[identifier]

    return count(node)


def graph_cost(roots: Sequence[Node]) -> tuple[int, int]:
    expanded = sum(operation_count(node) for node in roots)
    order, groups = layers(roots, {})
    if not groups:
        return expanded, expanded
    local = {id(node): 0 for node in order if isinstance(node, Col | Element)}
    cost = 0
    for index, group in enumerate(groups):
        variables = {**local, **{id(node): operation_count(node, local) for node in group}}
        required = needed(
            [*roots, *(node for later in groups[index + 1 :] for node in later)], variables
        )
        cost += 3 + 1 + sum(variables[key] for key in required)
        local = dict.fromkeys(required, 1)
    cost += 3 + 1 + 1 + sum(operation_count(node, local) for node in roots)
    return expanded, cost
