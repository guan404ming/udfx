from dataclasses import replace

from udfx.frontend.core.values import MAX_UNROLL
from udfx.ir import TRUE, Call, Comprehension, If, Lit, Node, lit
from udfx.ops.registry import apply
from udfx.types import STR, Kind


def simplify_list(node: Call) -> Node:
    if node.op.name == ".join":
        return join_filtered(node)
    if node.op.name != "__mul__" or node.type.kind is not Kind.LIST:
        return node
    value, count = node.arguments
    if value.type.kind is Kind.INT:
        value, count = count, value
    if not isinstance(count, Lit) or not isinstance(count.value, int) or count.value > MAX_UNROLL:
        return node
    if count.value <= 0:
        return Lit((), node.type)
    result = value
    for _ in range(count.value - 1):
        result = apply("__add__", result, value)
    return result


def join_filtered(node: Call) -> Node:
    separator, source = node.arguments
    if (
        separator != lit("")
        or not isinstance(source, Comprehension)
        or source.condition == TRUE
        or source.value.type != STR
    ):
        return node
    value = If(source.condition, source.value, lit(""), STR)
    return apply(".join", separator, replace(source, condition=TRUE, value=value))
