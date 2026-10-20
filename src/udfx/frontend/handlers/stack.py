from __future__ import annotations

import dis

from udfx.frontend.context import Context
from udfx.frontend.machine import handles
from udfx.frontend.state import State, Step


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
