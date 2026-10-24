from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from udfx.ir import Call, Lit, Node
from udfx.semantics.numbers import COMPARISONS

type Conditions = tuple[tuple[Node, bool], ...]
type Bounds = tuple[int, int]

REVERSED = {"__lt__": "__gt__", "__le__": "__ge__", "__gt__": "__lt__", "__ge__": "__le__"}
NEGATED = {
    "__lt__": "__ge__",
    "__le__": "__gt__",
    "__gt__": "__le__",
    "__ge__": "__lt__",
    "__eq__": "__ne__",
    "__ne__": "__eq__",
}
CONDITIONS: ContextVar[Conditions] = ContextVar("udfx_conditions", default=())


def predicates(node: Node, truth: bool) -> Conditions:
    if isinstance(node, Call):
        if node.op.name == "not":
            return predicates(node.arguments[0], not truth)
        if node.op.name == ("and" if truth else "or"):
            return tuple(
                part for argument in node.arguments for part in predicates(argument, truth)
            )
    return ((node, truth),)


@contextmanager
def assuming(node: Node, truth: bool) -> Generator[None]:
    token = CONDITIONS.set(implied(CONDITIONS.get() + predicates(node, truth)))
    try:
        yield
    finally:
        CONDITIONS.reset(token)


def known(node: Node, conditions: dict[Node, bool]) -> bool | None:
    if node in conditions:
        return conditions[node]
    if isinstance(node, Lit) and isinstance(node.value, bool):
        return node.value
    if isinstance(node, Call) and node.op.name == "not":
        truth = known(node.arguments[0], conditions)
        return None if truth is None else not truth
    return None


def implied(conditions: Conditions) -> Conditions:
    found = dict(conditions)
    while True:
        previous = len(found)
        for node, truth in tuple(found.items()):
            if not isinstance(node, Call) or node.op.name != ("or" if truth else "and"):
                continue
            left, right = node.arguments
            for argument, other in ((left, right), (right, left)):
                other_truth = known(other, found)
                if other_truth is not None and other_truth != truth:
                    found.update(predicates(argument, truth))
        if len(found) == previous:
            return tuple(found.items())


def constrain(bounds: Bounds, operation: str, other: Bounds) -> Bounds | None:
    lower, upper = bounds
    match operation:
        case "__lt__":
            upper = min(upper, other[1] - 1)
        case "__le__":
            upper = min(upper, other[1])
        case "__gt__":
            lower = max(lower, other[0] + 1)
        case "__ge__":
            lower = max(lower, other[0])
        case "__eq__":
            lower, upper = max(lower, other[0]), min(upper, other[1])
        case _:
            pass
    return (lower, upper) if lower <= upper else None


def compare(operation: str, left: Bounds, right: Bounds) -> bool | None:
    if operation in {"__eq__", "__ne__"}:
        if left[0] == left[1] == right[0] == right[1]:
            return operation == "__eq__"
        if left[1] < right[0] or right[1] < left[0]:
            return operation == "__ne__"
        return None
    comparator = COMPARISONS.get(operation)
    if comparator is None:
        return None
    results = {comparator(first, second) for first in left for second in right}
    return results.pop() if len(results) == 1 else None
