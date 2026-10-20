from __future__ import annotations

import dis
import operator

from udfx.errors import Unsupported
from udfx.frontend.calls import invoke, node_of, none_test, truth
from udfx.frontend.constants import GENERICS, annotation, lookup
from udfx.frontend.context import Context
from udfx.frontend.logic import negate
from udfx.frontend.machine import emit, handles
from udfx.frontend.state import State, Step
from udfx.frontend.values import Item, Opaque, Value
from udfx.ir import Lit, Node
from udfx.types import INT, Kind

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
        raise Unsupported("identity check against a value other than None", context.where(state))
    return [rest.push(Value(negate(test) if instruction.arg else test)).advance()]


@handles("CONTAINS_OP")
def contains_op(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item, container), rest = state.pop(2)
    members = container.node if isinstance(container, Value) else node_of(container)
    node, faults = invoke("__contains__", [members, node_of(item)], TypeError)
    return emit(context, rest, Value(negate(node) if instruction.arg else node), faults)


@handles("BINARY_SUBSCR")
def binary_subscr(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (container, index), rest = state.pop(2)
    if isinstance(container, Opaque) and any(container.obj is generic for generic in GENERICS):
        return [rest.push(annotation(operator.getitem, container, index)).advance()]
    if isinstance(container, Opaque):
        node, faults = lookup(container.obj, node_of(index))
    else:
        node, faults = invoke("__getitem__", [node_of(container), node_of(index)], TypeError)
    return emit(context, rest, Value(node), faults)


@handles("BINARY_SLICE")
def binary_slice(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (container, start, stop), rest = state.pop(3)
    node, faults = invoke("slice", [node_of(container), bound(start), bound(stop)], TypeError)
    return emit(context, rest, Value(node), faults)


def bound(item: Item) -> Node:
    node = node_of(item)
    return Lit(None, INT.or_none()) if node.type.kind is Kind.NONE else node


def is_none_constant(item: Item) -> bool:
    return isinstance(item, Value) and isinstance(item.node, Lit) and item.node.value is None
