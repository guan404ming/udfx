from udfx.ir import Call, Lit, Node, lit
from udfx.ops.registry import apply
from udfx.semantics.numbers.assumptions import NEGATED, REVERSED
from udfx.semantics.numbers.operations import FLOAT_INT_MAX, integer_comparison
from udfx.types import INT64_MAX, INT64_MIN, Kind

ARITHMETIC = frozenset(
    {"__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__", "__pow__"}
)


def float_literal(node: Node) -> Node:
    if not isinstance(node, Lit) or node.type.kind is not Kind.INT:
        return node
    value = node.value
    if not isinstance(value, int) or abs(value) > FLOAT_INT_MAX:
        return node
    return lit(float(value))


def widen_literals(node: Call) -> Node:
    if node.op.name not in NEGATED and node.op.name not in ARITHMETIC:
        return node
    if {argument.type.kind for argument in node.arguments} != {Kind.FLOAT, Kind.INT}:
        return node
    arguments = tuple(float_literal(argument) for argument in node.arguments)
    if arguments == node.arguments:
        return node
    return apply(node.op.name, *arguments)


IDENTITIES = {
    ("__add__", Kind.STR): ("", True),
    ("__add__", Kind.INT): (0, True),
    ("__sub__", Kind.INT): (0, False),
    ("__mul__", Kind.INT): (1, True),
    ("__mul__", Kind.FLOAT): (1.0, True),
    ("__truediv__", Kind.FLOAT): (1.0, False),
}


def identity(node: Call) -> Node:
    found = IDENTITIES.get((node.op.name, node.type.kind))
    if found is None:
        return node
    value, commutative = found
    left, right = node.arguments
    if right == lit(value) and left.type == node.type:
        return left
    if commutative and left == lit(value) and right.type == node.type:
        return right
    return node


def simplify_number(node: Call) -> Node:
    operation = node.op.name
    if operation not in NEGATED:
        return node
    left, right = node.arguments
    if isinstance(left, Lit) and (
        isinstance(left.value, float)
        or (
            isinstance(right, Call)
            and right.op.name == "int"
            and right.arguments[0].type.kind is Kind.STR
        )
    ):
        left, right = right, left
        operation = REVERSED.get(operation, operation)
    if left.type.kind is not Kind.INT or not isinstance(right, Lit):
        return node
    if isinstance(right.value, float):
        normalized = integer_comparison(operation, right.value)
        if isinstance(normalized, bool):
            return lit(normalized)
        operation, value = normalized
        right = lit(value)
    if right.type.kind is not Kind.INT:
        return node
    if (
        isinstance(left, Call)
        and left.op.name == "int"
        and left.arguments[0].type.kind is Kind.STR
        and isinstance(right.value, int)
        and INT64_MIN < right.value < INT64_MAX
    ):
        left = apply("int_saturated", *left.arguments)
    return apply(operation, left, right)
