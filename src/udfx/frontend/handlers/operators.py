from __future__ import annotations

import dis
import operator

from udfx.errors import Unsupported
from udfx.frontend.calls import invoke, node_of, none_test, truth
from udfx.frontend.constants import GENERICS, annotation, constant, lookup, type_comparison
from udfx.frontend.context import Context
from udfx.frontend.formatting import format_percent
from udfx.frontend.logic import fold, negate
from udfx.frontend.machine import emit, handles
from udfx.frontend.state import State, Step
from udfx.frontend.values import Item, Opaque, Row, Tuple, Value
from udfx.ir import Lit, lit
from udfx.types import Kind

BINARY_OPS: dict[str, str] = {
    "+": "__add__",
    "-": "__sub__",
    "*": "__mul__",
    "/": "__truediv__",
    "//": "__floordiv__",
    "%": "__mod__",
    "**": "__pow__",
}

COMPARE_OPS: dict[str, str] = {
    "<": "__lt__",
    "<=": "__le__",
    "==": "__eq__",
    "!=": "__ne__",
    ">": "__gt__",
    ">=": "__ge__",
}


@handles("BINARY_OP")
def binary_op(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    symbol = instruction.argrepr.removesuffix("=")
    (left, right), rest = state.pop(2)
    if symbol == "|":
        return [rest.push(annotation(operator.or_, left, right)).advance()]
    if symbol == "%" and isinstance(left, Value) and left.node.type.kind is Kind.STR:
        if isinstance(right, Value) and right.container is not tuple:
            node_of(right)
        match right:
            case Tuple(items=items):
                arguments = items
            case Value(node=Lit(value=tuple() as values)) | Opaque(value=tuple() as values):
                arguments = tuple(constant(value) for value in values)
            case _:
                arguments = (right,)
        node, faults = format_percent(left.node, [node_of(item) for item in arguments])
        return emit(context, rest, Value(fold(node)), faults)
    name = BINARY_OPS.get(symbol)
    if name is None:
        raise Unsupported(f"binary operator {instruction.argrepr}", context.where(state))
    node, faults = invoke(name, [node_of(left), node_of(right)], TypeError)
    return emit(context, rest, Value(node), faults)


@handles("COMPARE_OP")
def compare_op(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    name = COMPARE_OPS.get(str(instruction.argval))
    if name is None:
        raise Unsupported(f"comparison {instruction.argval}", context.where(state))
    (left, right), rest = state.pop(2)
    if name in ("__eq__", "__ne__"):
        test = type_comparison(left, right)
        if test is not None:
            return [rest.push(Value(negate(test) if name == "__ne__" else test)).advance()]
    node, faults = invoke(name, [node_of(left), node_of(right)], TypeError)
    return emit(context, rest, Value(node), faults)


@handles("UNARY_NOT")
def unary_not(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    test, faults = truth(item)
    return emit(context, rest, Value(negate(test)), faults)


@handles("UNARY_NEGATIVE")
def unary_negative(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    node, faults = invoke("__neg__", [node_of(item)], TypeError)
    return emit(context, rest, Value(node), faults)


@handles("IS_OP")
def is_op(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (left, right), rest = state.pop(2)
    if is_none_constant(right):
        test = none_test(left)
    elif is_none_constant(left):
        test = none_test(right)
    else:
        test = type_comparison(left, right)
        if test is None:
            raise Unsupported(
                "identity check against a value other than None", context.where(state)
            )
    return [rest.push(Value(negate(test) if instruction.arg else test)).advance()]


@handles("CONTAINS_OP")
def contains_op(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item, container), rest = state.pop(2)
    if isinstance(container, Opaque) and isinstance(container.value, dict):
        node, faults = lookup(dict.fromkeys(container.value, True), node_of(item), lit(False))
        return emit(context, rest, Value(negate(node) if instruction.arg else node), faults)
    members = container.node if isinstance(container, Value) else node_of(container)
    node, faults = invoke("__contains__", [members, node_of(item)], TypeError)
    return emit(context, rest, Value(negate(node) if instruction.arg else node), faults)


@handles("BINARY_SUBSCR")
def binary_subscr(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (container, index), rest = state.pop(2)
    if isinstance(container, Row):
        if not (
            isinstance(index, Value)
            and isinstance(index.node, Lit)
            and isinstance(index.node.value, str)
        ):
            raise Unsupported("row access requires a constant string key", context.where(state))
        return [rest.push(container.column(index.node.value)).advance()]
    if isinstance(container, Opaque) and any(container.value is generic for generic in GENERICS):
        return [rest.push(annotation(operator.getitem, container, index)).advance()]
    if isinstance(container, Opaque):
        node, faults = lookup(container.value, node_of(index))
    else:
        node, faults = invoke("__getitem__", [node_of(container), node_of(index)], TypeError)
    return emit(context, rest, Value(node), faults)


@handles("BINARY_SLICE")
def binary_slice(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (container, start, stop), rest = state.pop(3)
    node, faults = invoke("slice", [node_of(container), node_of(start), node_of(stop)], TypeError)
    return emit(context, rest, Value(node), faults)


def is_none_constant(item: Item) -> bool:
    return isinstance(item, Value) and isinstance(item.node, Lit) and item.node.value is None
