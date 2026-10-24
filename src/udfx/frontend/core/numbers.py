from udfx.ir import Call, Lit, Node, lit
from udfx.ops.registry import apply
from udfx.semantics.assumptions import NEGATED, REVERSED
from udfx.semantics.numbers import integer_comparison
from udfx.types import INT64_MAX, INT64_MIN, Kind


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
