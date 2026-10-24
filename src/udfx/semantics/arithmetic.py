from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from udfx.ir import Call, Node
from udfx.semantics.numbers import ARITHMETIC
from udfx.semantics.ranges import integer_bounds
from udfx.types import INT64_MAX, INT64_MIN, Kind


@dataclass(frozen=True)
class IntegerRegionPrimitives:
    maximum: int
    size_limit: int
    widen: Callable[[Any], Any]
    narrow: Callable[[Any], Any]
    not_null: Callable[[Any], Any]
    operations: Mapping[str, Callable[[Any, Any], Any]]


def integer_arithmetic(node: Node) -> bool:
    return (
        isinstance(node, Call)
        and node.op.name in ARITHMETIC
        and all(argument.type.kind is Kind.INT for argument in node.arguments)
    )


def integer_region(node: Node, maximum: int, size_limit: int) -> frozenset[Node]:
    bounds: dict[Node, tuple[int, int]] = {}
    operations: set[Node] = set()

    def visit(part: Node) -> bool:
        if part in bounds:
            return True
        if isinstance(part, Call) and integer_arithmetic(part):
            if not all(visit(argument) for argument in part.arguments):
                return False
            left, right = (bounds[argument] for argument in part.arguments)
            values = [ARITHMETIC[part.op.name](first, second) for first in left for second in right]
            found = min(values), max(values)
            operations.add(part)
        else:
            found = integer_bounds(part, size_limit)
        if found is None or not -maximum <= found[0] <= found[1] <= maximum:
            return False
        bounds[part] = found
        return True

    if not integer_arithmetic(node) or not visit(node):
        return frozenset()
    intermediate = operations - {node}
    needed = any(
        bounds[part][0] < INT64_MIN or bounds[part][1] > INT64_MAX for part in intermediate
    )
    return frozenset(operations) if needed else frozenset()
