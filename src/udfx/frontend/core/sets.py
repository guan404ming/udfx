from udfx.ir import Call, If, Node, lit
from udfx.ops.registry import apply
from udfx.semantics.lists import SET_OPERATIONS, SET_PREDICATES
from udfx.semantics.ranges import proven
from udfx.types import BOOL

SET_CALLS = frozenset(f"set_{operation}" for operation in SET_OPERATIONS)
SET_CONSUMERS = SET_CALLS | {
    *(f"set_{operation}_size" for operation in SET_OPERATIONS),
    *(f"set_{operation}" for operation in SET_PREDICATES),
}


def union_items(node: Node) -> Node:
    if isinstance(node, Call) and node.op.name == "set_union":
        return apply("__add__", *(union_items(argument) for argument in node.arguments))
    return node


def empty_union(node: Call) -> Node:
    left, right = node.arguments
    if left == lit(0):
        left, right = right, left
    if right != lit(0) or not isinstance(left, Call) or left.op.name != "set_union_size":
        return node
    empty = apply("and", *(apply("__eq__", apply("len", part), lit(0)) for part in left.arguments))
    return empty if node.op.name == "__eq__" else apply("not", empty)


def disjoint_ratio(node: Call) -> Node | None:
    numerator, denominator = node.arguments
    if isinstance(numerator, Call) and numerator.op.name == "float":
        numerator = numerator.arguments[0]
    if (
        not isinstance(numerator, Call)
        or numerator.op.name != "set_intersection_size"
        or not isinstance(denominator, Call)
        or denominator.op.name != "set_union_size"
        or numerator.arguments != denominator.arguments
    ):
        return None
    empty = apply("__eq__", numerator, lit(0))
    return If(empty, lit(0.0), node, node.type) if proven(empty) is None else None


def simplify_set(node: Call) -> Node:
    if node.op.name == "set_difference":
        left, right = node.arguments
        if (
            isinstance(left, Call)
            and left.op.name == "set_union"
            and isinstance(right, Call)
            and right.op.name == "set_intersection"
            and left.arguments in (right.arguments, right.arguments[::-1])
        ):
            return apply("set_symmetric_difference", *left.arguments)
    if node.op.name in {"sorted", "sorted_set"}:
        source = node.arguments[0]
        if isinstance(source, Call) and source.op.name in SET_CALLS:
            return source
    if node.op.name == "set_size":
        source = node.arguments[0]
        if isinstance(source, Call) and source.op.name in SET_CALLS:
            return apply(f"{source.op.name}_size", *source.arguments)
    if node.op.name != "__contains__":
        return (
            Call(node.op, tuple(union_items(argument) for argument in node.arguments))
            if node.op.name in SET_CONSUMERS
            else node
        )
    return set_membership(node)


def set_membership(node: Call) -> Node:
    source, item = node.arguments
    if not isinstance(source, Call) or source.op.name not in SET_CALLS:
        return node
    left, right = (
        simplify_set(apply("__contains__", argument, item)) for argument in source.arguments
    )
    match source.op.name:
        case "set_union":
            return apply("or", left, right)
        case "set_intersection":
            return apply("and", left, right)
        case "set_difference":
            return apply("and", left, apply("not", right))
        case _:
            return If(left, apply("not", right), right, BOOL)
