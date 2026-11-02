import math

from udfx.ir import Call, If, Lit, Node
from udfx.semantics.assumptions import CONDITIONS, REVERSED
from udfx.types import Kind

type Interval = tuple[float, float, bool]

UNBOUNDED: Interval = (-math.inf, math.inf, True)
INTRINSIC: dict[str, Interval] = {
    "math.atan2": (-math.pi, math.pi, True),
    "math.sin": (-1.0, 1.0, True),
    "math.cos": (-1.0, 1.0, True),
    "math.sqrt": (0.0, math.inf, True),
    "math.exp": (0.0, math.inf, True),
}
COMPARISONS = {
    "__lt__": lambda below, above, low, high: (below, low),
    "__le__": lambda below, above, low, high: (high, above),
    "__gt__": lambda below, above, low, high: (above, high),
    "__ge__": lambda below, above, low, high: (low, below),
}


def float_fact(node: Call) -> bool | None:
    if any(argument.type.kind is not Kind.FLOAT for argument in node.arguments):
        return None
    seen: dict[int, Interval] = {}
    if node.op.name in {"math.isinf", "math.isnan"}:
        low, high, nan = float_bounds(node.arguments[0], seen)
        bounded = not nan if node.op.name == "math.isnan" else -math.inf < low and high < math.inf
        return False if bounded else None
    if node.op.name not in {*COMPARISONS, "__eq__", "__ne__"}:
        return None
    (first_low, first_high, first_nan), (second_low, second_high, second_nan) = (
        float_bounds(argument, seen) for argument in node.arguments
    )
    below, above = first_high < second_low, first_low > second_high
    if node.op.name in {"__eq__", "__ne__"}:
        disjoint = below or above
        return (node.op.name == "__ne__") if disjoint else None
    ordered = not first_nan and not second_nan
    true, false = COMPARISONS[node.op.name](
        below, above, first_low >= second_high, first_high <= second_low
    )
    return True if ordered and true else False if false else None


def float_bounds(node: Node, seen: dict[int, Interval]) -> Interval:
    if id(node) not in seen:
        bounds = intrinsic_bounds(node, seen)
        for predicate, truth in CONDITIONS.get():
            if truth and isinstance(predicate, Call) and predicate.op.name in REVERSED:
                bounds = constrained(bounds, node, predicate)
        seen[id(node)] = bounds
    return seen[id(node)]


def constrained(bounds: Interval, node: Node, predicate: Call) -> Interval:
    left, right = predicate.arguments
    operation = predicate.op.name
    if right == node and isinstance(left, Lit):
        left, right, operation = right, left, REVERSED[operation]
    if left != node or not isinstance(right, Lit) or not isinstance(right.value, float):
        return bounds
    low, high, _ = bounds
    if operation in {"__gt__", "__ge__"}:
        low = max(low, right.value)
    else:
        high = min(high, right.value)
    return low, high, False


def intrinsic_bounds(node: Node, seen: dict[int, Interval]) -> Interval:
    if isinstance(node, Lit) and isinstance(node.value, float):
        return UNBOUNDED if math.isnan(node.value) else (node.value, node.value, False)
    if isinstance(node, If) and node.type.kind is Kind.FLOAT:
        first, second = float_bounds(node.then, seen), float_bounds(node.orelse, seen)
        return min(first[0], second[0]), max(first[1], second[1]), first[2] or second[2]
    if not isinstance(node, Call) or node.type.kind is not Kind.FLOAT:
        return UNBOUNDED
    if any(argument.type.kind is not Kind.FLOAT for argument in node.arguments):
        return UNBOUNDED
    if node.op.name in INTRINSIC:
        return INTRINSIC[node.op.name]
    return call_bounds(node, [float_bounds(argument, seen) for argument in node.arguments])


def call_bounds(node: Call, bounds: list[Interval]) -> Interval:
    name = node.op.name
    if name == "abs":
        low, high, nan = bounds[0]
        return 0.0, max(abs(low), abs(high)), nan
    if name == "__pow__":
        exponent = node.arguments[1]
        even = (
            isinstance(exponent, Lit)
            and isinstance(exponent.value, float)
            and exponent.value % 2.0 == 0.0
        )
        return (0.0, math.inf, bounds[0][2]) if even else UNBOUNDED
    if name in {"min", "max"}:
        return extremum(name, *bounds)
    if name not in {"__add__", "__sub__", "__mul__", "__truediv__"}:
        return UNBOUNDED
    (first_low, first_high, first_nan), (second_low, second_high, second_nan) = bounds
    if name == "__truediv__":
        if second_low <= 0.0 <= second_high:
            return UNBOUNDED
        quotients = [
            first / second
            for first in (first_low, first_high)
            for second in (second_low, second_high)
        ]
        if any(math.isnan(quotient) for quotient in quotients):
            return UNBOUNDED
        return min(quotients), max(quotients), first_nan or second_nan
    if name == "__sub__":
        second_low, second_high = -second_high, -second_low
    if name == "__mul__":
        corners = [
            first * second
            for first in (first_low, first_high)
            for second in (second_low, second_high)
        ]
        undefined = (
            first_low <= 0.0 <= first_high and math.inf in map(abs, (second_low, second_high))
        ) or (second_low <= 0.0 <= second_high and math.inf in map(abs, (first_low, first_high)))
    else:
        corners = [first_low + second_low, first_high + second_high]
        undefined = math.inf in (first_high, second_high) and -math.inf in (first_low, second_low)
    if any(math.isnan(corner) for corner in corners):
        return UNBOUNDED
    return min(corners), max(corners), first_nan or second_nan or undefined


def extremum(name: str, first: Interval, second: Interval) -> Interval:
    pick = min if name == "min" else max
    low, high = pick(first[0], second[0]), pick(first[1], second[1])
    if name == "min":
        low = min(first[0], second[0])
    else:
        high = max(first[1], second[1])
    if second[2]:
        low, high = min(low, first[0]), max(high, first[1])
    return low, high, first[2]


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
