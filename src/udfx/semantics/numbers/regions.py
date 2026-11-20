from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

from udfx.ir import Call, Lit, Node, children
from udfx.semantics.bindings import source
from udfx.semantics.numbers.operations import ARITHMETIC
from udfx.semantics.numbers.ranges import integer_bounds
from udfx.types import INT64_MAX, INT64_MIN, Kind


@dataclass(frozen=True)
class IntegerRegion:
    operations: frozenset[Node]
    narrow: bool


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


def integer_dependencies(node: Node) -> Iterator[Node]:
    pending = [node]
    seen: set[int] = set()
    while pending:
        part = pending.pop()
        if id(part) in seen:
            continue
        seen.add(id(part))
        if part is node or (part.type.kind is Kind.INT and part != Lit(None, part.type)):
            yield part
        pending.extend(children(source(part)))


def escapes(part: Call, bounds: Mapping[Node, tuple[int, int]]) -> bool:
    operation = ARITHMETIC[part.op.name]
    for index, argument in enumerate(part.arguments):
        other = bounds[part.arguments[1 - index]]
        low, high = bounds[argument]
        for overflow in ((max(low, INT64_MAX + 1), high), (low, min(high, INT64_MIN - 1))):
            if overflow[0] > overflow[1]:
                continue
            pairs = product(overflow, other) if index == 0 else product(other, overflow)
            values = [operation(first, second) for first, second in pairs]
            if min(values) <= INT64_MAX and max(values) >= INT64_MIN:
                return False
    return True


def integer_region(node: Node, maximum: int, size_limit: int) -> IntegerRegion | None:
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
        return None
    intermediate = operations - {node}
    if not any(bounds[part][0] < INT64_MIN or bounds[part][1] > INT64_MAX for part in intermediate):
        return None
    narrow = all(escapes(part, bounds) for part in operations if isinstance(part, Call))
    return IntegerRegion(frozenset(operations), narrow)
