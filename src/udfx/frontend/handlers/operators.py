from __future__ import annotations

import dis

from udfx.errors import Unsupported
from udfx.frontend.calls import invoke, node_of
from udfx.frontend.context import Context
from udfx.frontend.machine import emit, handles
from udfx.frontend.state import State, Step
from udfx.frontend.values import Value

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
