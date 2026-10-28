from __future__ import annotations

import operator
from collections.abc import Callable

from udfx.ir import Call, Col, Comprehension, Element, If, ListValue, Lit, Node
from udfx.ops.registry import apply
from udfx.semantics.assumptions import (
    CONDITIONS,
    NEGATED,
    REVERSED,
    Conditions,
    compare,
    constrain,
    predicates,
)
from udfx.semantics.bindings import SOURCES, source
from udfx.semantics.lists import SET_OPERATIONS
from udfx.types import INT64_MAX, INT64_MIN, Kind

LENGTH_OPS = frozenset(
    {
        "len",
        "set_size",
        "set_size_float",
        *(f"set_{operation}_size" for operation in SET_OPERATIONS),
    }
)


OPERATIONS: dict[str, Callable[[int, int], int]] = {
    "__add__": operator.add,
    "__sub__": operator.sub,
    "__mul__": operator.mul,
    "__floordiv__": operator.floordiv,
    "min": min,
    "max": max,
}


def maximum_length(node: Node) -> int | None:
    node = source(node)
    if isinstance(node, Lit) and isinstance(node.value, str):
        return len(node.value)
    if not isinstance(node, Call):
        return None
    if node.op.name in {".strip", ".lstrip", ".rstrip"}:
        return maximum_length(node.arguments[0])
    if node.op.name != "slice":
        return None
    text, start, stop, *steps = node.arguments
    if steps and (not isinstance(steps[0], Lit) or steps[0].value not in (None, 1)):
        return maximum_length(text)
    bounds = [maximum_length(text)]
    if isinstance(start, Lit) and isinstance(start.value, int) and start.value < 0:
        bounds.append(-start.value)
    if isinstance(stop, Lit) and isinstance(stop.value, int) and stop.value >= 0:
        begin = start.value if isinstance(start, Lit) and isinstance(start.value, int) else 0
        bounds.append(max(stop.value - max(begin, 0), 0))
    known = [bound for bound in bounds if bound is not None]
    return min(known) if known else None


def integer_bounds(
    node: Node, size_limit: int, conditions: Conditions | None = None
) -> tuple[int, int] | None:
    conditions = CONDITIONS.get() if conditions is None else conditions
    bindings = SOURCES.get()
    if bindings is not None and isinstance(node, Element) and node in bindings.values:
        key = node, size_limit
        if key not in bindings.ranges:
            bindings.ranges[key] = integer_bounds(bindings.values[node], size_limit, ())
        bounds = bindings.ranges[key]
    else:
        bounds = intrinsic_bounds(node, size_limit, conditions)
    for predicate, truth in conditions:
        if bounds is None or not isinstance(predicate, Call) or predicate.op.name not in NEGATED:
            continue
        left, right = predicate.arguments
        operation = predicate.op.name
        if node == left:
            other = right
        elif node == right:
            other = left
            operation = REVERSED.get(operation, operation)
        else:
            continue
        compared = integer_bounds(other, size_limit, ())
        if compared is not None:
            bounds = constrain(bounds, operation if truth else NEGATED[operation], compared)
    return bounds


def intrinsic_bounds(node: Node, size_limit: int, conditions: Conditions) -> tuple[int, int] | None:
    if node.type.kind not in (Kind.INT, Kind.BOOL):
        return None
    if isinstance(node, Lit):
        return (int(node.value), int(node.value)) if isinstance(node.value, int) else None
    if node.type.kind is Kind.BOOL:
        return 0, 1
    if isinstance(node, Col | Element):
        return INT64_MIN, INT64_MAX
    if isinstance(node, If):
        left = integer_bounds(node.then, size_limit, conditions + predicates(node.condition, True))
        right = integer_bounds(
            node.orelse, size_limit, conditions + predicates(node.condition, False)
        )
        return combine(left, right)
    if not isinstance(node, Call):
        return None
    return call_bounds(node, size_limit, conditions)


def call_bounds(node: Call, size_limit: int, conditions: Conditions) -> tuple[int, int] | None:
    if node.op.name in LENGTH_OPS:
        return (
            split_length(node.arguments[0], size_limit, conditions)
            if node.op.name == "len"
            else None
        ) or (0, size_limit)
    if node.op.name in {".find", ".rfind"}:
        return -1, size_limit
    if node.op.name == ".count":
        return 0, size_limit + 1
    if node.op.name == "__getitem__":
        return element_bounds(node.arguments[0], size_limit, conditions)
    if node.op.name == "__mod__":
        divisor = node.arguments[1]
        if isinstance(divisor, Lit) and isinstance(divisor.value, int) and divisor.value:
            return (0, divisor.value - 1) if divisor.value > 0 else (divisor.value + 1, 0)
        return None
    if node.op.name == "int" and node.arguments[0].type.kind is Kind.STR:
        length = maximum_length(node.arguments[0])
        if length is not None and length <= 18:
            maximum = 10**length - 1
            return -maximum, maximum
        return None
    if node.op.name == "sum":
        bounds = element_bounds(node.arguments[0], size_limit, conditions)
        return (
            (min(0, bounds[0] * size_limit), max(0, bounds[1] * size_limit))
            if bounds is not None
            else None
        )
    operation = OPERATIONS.get(node.op.name)
    if operation is None:
        return None
    left, right = (integer_bounds(argument, size_limit, conditions) for argument in node.arguments)
    if left is None or right is None:
        return None
    if node.op.name == "__floordiv__" and right[0] <= 0 <= right[1]:
        return None
    values = [operation(first, second) for first in left for second in right]
    lower, upper = min(values), max(values)
    return (lower, upper) if INT64_MIN <= lower <= upper <= INT64_MAX else None


def split_length(node: Node, size_limit: int, conditions: Conditions) -> tuple[int, int] | None:
    if not isinstance(node, Call) or node.op.name != ".split" or len(node.arguments) != 2:
        return None
    source, separator = node.arguments
    if (
        not isinstance(separator, Lit)
        or not isinstance(separator.value, str)
        or not separator.value
    ):
        return None
    count = integer_bounds(apply(".count", source, separator), size_limit, conditions)
    return (count[0] + 1, min(count[1] + 1, size_limit)) if count is not None else None


def combine(left: tuple[int, int] | None, right: tuple[int, int] | None) -> tuple[int, int] | None:
    return (min(left[0], right[0]), max(left[1], right[1])) if left and right else None


def element_bounds(node: Node, size_limit: int, conditions: Conditions) -> tuple[int, int] | None:
    node = source(node)
    if isinstance(node, Comprehension):
        return integer_bounds(node.value, size_limit, conditions)
    if isinstance(node, Lit) and isinstance(node.value, tuple | list):
        if all(isinstance(value, int) for value in node.value):
            return min(node.value, default=0), max(node.value, default=0)
        return None
    if isinstance(node, ListValue):
        bounds = (0, 0)
        for item in node.items:
            found = combine(bounds, integer_bounds(item, size_limit, conditions))
            if found is None:
                return None
            bounds = found
        return bounds
    if isinstance(node, If):
        return combine(
            element_bounds(node.then, size_limit, conditions + predicates(node.condition, True)),
            element_bounds(node.orelse, size_limit, conditions + predicates(node.condition, False)),
        )
    if isinstance(node, Call) and node.op.name in {"slice", "list"}:
        return element_bounds(node.arguments[0], size_limit, conditions)
    if node.type.item is not None:
        if node.type.item.kind is Kind.INT:
            return INT64_MIN, INT64_MAX
        if node.type.item.kind is Kind.BOOL:
            return 0, 1
    return None


def proven(node: Node) -> bool | None:
    for predicate, truth in CONDITIONS.get():
        if node == predicate:
            return truth
    if not isinstance(node, Call) or node.op.name not in NEGATED:
        return None
    constant = constant_comparison(node)
    if constant is not None:
        return constant
    left, right = (integer_bounds(argument, INT64_MAX) for argument in node.arguments)
    return compare(node.op.name, left, right) if left is not None and right is not None else None


def constant_comparison(node: Call) -> bool | None:
    left, right = node.arguments
    for index, argument in enumerate(node.arguments):
        if isinstance(argument, If) and isinstance(node.arguments[1 - index], Lit):
            results = [
                constant_comparison(
                    Call(node.op, (branch, right) if index == 0 else (left, branch))
                )
                for branch in (argument.then, argument.orelse)
            ]
            return results[0] if results[0] is not None and results[0] == results[1] else None
    if not isinstance(left, Lit) or not isinstance(right, Lit):
        return None
    try:
        result = node.op.reference(left.value, right.value)
        return result if isinstance(result, bool) else None
    except (TypeError, ValueError):
        return None
