from __future__ import annotations

import dis

from udfx.errors import Unsupported
from udfx.frontend.calls import invoke, node_of, none_test, truth
from udfx.frontend.context import Context
from udfx.frontend.logic import neg
from udfx.frontend.machine import emit, handles
from udfx.frontend.state import State, Step
from udfx.frontend.values import Item, Value
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
def binary_op(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    symbol = inst.argrepr.removesuffix("=")
    name = BINARY_OPS.get(symbol)
    if name is None:
        raise Unsupported(f"binary operator {inst.argrepr}", ctx.where(state))
    (lhs, rhs), rest = state.pop(2)
    node, faults = invoke(name, [node_of(lhs), node_of(rhs)], TypeError)
    return emit(ctx, rest, Value(node), faults)


@handles("COMPARE_OP")
def compare_op(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    name = COMPARE_OPS.get(str(inst.argval))
    if name is None:
        raise Unsupported(f"comparison {inst.argval}", ctx.where(state))
    (lhs, rhs), rest = state.pop(2)
    node, faults = invoke(name, [node_of(lhs), node_of(rhs)], TypeError)
    return emit(ctx, rest, Value(node), faults)


@handles("UNARY_NOT")
def unary_not(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    test, faults = truth(item)
    return emit(ctx, rest, Value(neg(test)), faults)


@handles("UNARY_NEGATIVE")
def unary_negative(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    node, faults = invoke("__neg__", [node_of(item)], TypeError)
    return emit(ctx, rest, Value(node), faults)


@handles("IS_OP")
def is_op(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    (lhs, rhs), rest = state.pop(2)
    if is_none_constant(rhs):
        test = none_test(lhs)
    elif is_none_constant(lhs):
        test = none_test(rhs)
    else:
        raise Unsupported("identity check against a value other than None", ctx.where(state))
    return [rest.push(Value(neg(test) if inst.arg else test)).advance()]


@handles("CONTAINS_OP")
def contains_op(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    (item, container), rest = state.pop(2)
    node, faults = invoke("__contains__", [node_of(container), node_of(item)], TypeError)
    return emit(ctx, rest, Value(neg(node) if inst.arg else node), faults)


@handles("BINARY_SUBSCR")
def binary_subscr(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    (container, index), rest = state.pop(2)
    node, faults = invoke("__getitem__", [node_of(container), node_of(index)], TypeError)
    return emit(ctx, rest, Value(node), faults)


@handles("BINARY_SLICE")
def binary_slice(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    (container, start, stop), rest = state.pop(3)
    node, faults = invoke("slice", [node_of(container), bound(start), bound(stop)], TypeError)
    return emit(ctx, rest, Value(node), faults)


def bound(item: Item) -> Node:
    node = node_of(item)
    return Lit(None, INT.or_none()) if node.type.kind is Kind.NONE else node


def is_none_constant(item: Item) -> bool:
    return isinstance(item, Value) and isinstance(item.node, Lit) and item.node.value is None
