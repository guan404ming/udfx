from __future__ import annotations

import dis

from udfx.errors import Unsupported
from udfx.frontend.calls import call
from udfx.frontend.context import Context
from udfx.frontend.machine import emit, handles
from udfx.frontend.state import State, Step
from udfx.frontend.values import NULL, Method, Null, Value


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
