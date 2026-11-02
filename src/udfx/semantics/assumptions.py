from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from udfx.ir import Call, If, Lit, Node
from udfx.semantics.numbers import COMPARISONS
from udfx.types import BOOL

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
    found: list[tuple[Node, bool]] = []
    pending = [(node, truth)]
    seen: set[tuple[int, bool]] = set()
    while pending:
        current, positive = pending.pop()
        key = id(current), positive
        if key in seen:
            continue
        seen.add(key)
        if isinstance(current, If) and current.type == current.condition.type == BOOL:
            found.append((current, positive))
            pending.extend(reversed(branch_predicates(current, positive, {})))
        elif isinstance(current, Call) and current.op.name == "not":
            pending.append((current.arguments[0], not positive))
        elif isinstance(current, Call) and current.op.name == ("and" if positive else "or"):
            pending.extend((argument, positive) for argument in reversed(current.arguments))
        else:
            found.append((current, positive))
    return tuple(found)


@contextmanager
def assuming(node: Node, truth: bool) -> Generator[None]:
    token = CONDITIONS.set(implied(CONDITIONS.get() + predicates(node, truth)))
    try:
        yield
    finally:
        CONDITIONS.reset(token)


def known(node: Node, conditions: dict[Node, bool]) -> bool | None:
    seen: dict[int, bool | None] = {}

    def visit(value: Node) -> bool | None:
        if value in conditions:
            return conditions[value]
        if id(value) in seen:
            return seen[id(value)]
        result = None
        if isinstance(value, Lit) and isinstance(value.value, bool):
            result = value.value
        elif isinstance(value, Call) and value.op.name == "not":
            truth = visit(value.arguments[0])
            result = None if truth is None else not truth
        elif isinstance(value, If) and value.type == value.condition.type == BOOL:
            truth = visit(value.condition)
            if truth is not None:
                result = visit(value.then if truth else value.orelse)
            else:
                left, right = visit(value.then), visit(value.orelse)
                result = left if left == right else None
        seen[id(value)] = result
        return result

    return visit(node)


def branch_predicates(node: If, truth: bool, conditions: dict[Node, bool]) -> Conditions:
    selected = known(node.condition, conditions)
    if selected is None:
        for branch, positive in ((node.then, True), (node.orelse, False)):
            branch_truth = known(branch, conditions)
            if branch_truth is not None and branch_truth != truth:
                selected = not positive
                break
    if selected is None:
        return ()
    branch = node.then if selected else node.orelse
    return (node.condition, selected), (branch, truth)


def implied(conditions: Conditions) -> Conditions:
    found = dict(conditions)
    while True:
        previous = len(found)
        for node, truth in tuple(found.items()):
            if isinstance(node, If) and node.type == node.condition.type == BOOL:
                found.update(
                    fact
                    for branch, positive in branch_predicates(node, truth, found)
                    for fact in predicates(branch, positive)
                )
                continue
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
