from __future__ import annotations

import dis
from collections.abc import Callable, Sequence

from udfx.errors import Unsupported
from udfx.frontend.calls import call, invoke, node_of
from udfx.frontend.context import Context
from udfx.frontend.state import Return, State, Step
from udfx.frontend.values import NULL, Method, Null, Value
from udfx.ops import Fault

Handler = Callable[[Context, State, dis.Instruction], list[Step]]

HANDLERS: dict[str, Handler] = {}

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


def handles(*opnames: str) -> Callable[[Handler], Handler]:
    def register(handler: Handler) -> Handler:
        for opname in opnames:
            if opname in HANDLERS:
                raise ValueError(f"duplicate handler {opname}")
            HANDLERS[opname] = handler
        return handler

    return register


def emit(ctx: Context, state: State, result: Value, faults: Sequence[Fault]) -> list[Step]:
    survivor, steps = ctx.guard(state, faults)
    if survivor is not None:
        steps.append(survivor.push(result).advance())
    return steps


@handles("RESUME", "NOP", "CACHE")
def skip(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    return [state.advance()]


@handles("LOAD_FAST", "LOAD_FAST_CHECK")
def load_fast(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    return [state.push(state.load(inst.argval)).advance()]


@handles("STORE_FAST")
def store_fast(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    return [rest.store(inst.argval, item).advance()]


@handles("LOAD_CONST")
def load_const(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    return [state.push(ctx.constant(inst.argval)).advance()]


@handles("POP_TOP")
def pop_top(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    return [state.pop()[1].advance()]


@handles("PUSH_NULL")
def push_null(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    return [state.push(NULL).advance()]


@handles("LOAD_GLOBAL")
def load_global(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    item = ctx.global_(inst.argval)
    pushed = state.push(NULL, item) if inst.arg is not None and inst.arg & 1 else state.push(item)
    return [pushed.advance()]


@handles("LOAD_ATTR")
def load_attr(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    if inst.arg is None or not inst.arg & 1:
        raise Unsupported(f"attribute load {inst.argval}", ctx.where(state))
    (owner,), rest = state.pop()
    return [rest.push(Method(inst.argval), owner).advance()]


@handles("CALL")
def call_(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    argc = inst.arg or 0
    items, rest = state.pop(argc + 2)
    head, second, *args = items
    callee, arguments = (second, args) if isinstance(head, Null) else (head, [second, *args])
    node, faults = call(callee, arguments)
    return emit(ctx, rest, Value(node), faults)


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


@handles("RETURN_VALUE")
def return_value(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    (item,), _ = state.pop()
    return [Return(state.path, node_of(item))]


@handles("RETURN_CONST")
def return_const(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    return [Return(state.path, node_of(ctx.constant(inst.argval)))]
