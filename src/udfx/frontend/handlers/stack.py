from __future__ import annotations

import dis

from udfx.frontend.context import Context, constant
from udfx.frontend.machine import handles
from udfx.frontend.state import State, Step


@handles("RESUME", "NOP", "CACHE")
def skip(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.advance()]


@handles("LOAD_FAST", "LOAD_FAST_CHECK")
def load_fast(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.push(state.load(instruction.argval)).advance()]


@handles("STORE_FAST")
def store_fast(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    return [rest.store(instruction.argval, item).advance()]


@handles("LOAD_CONST")
def load_const(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.push(constant(instruction.argval)).advance()]


@handles("POP_TOP")
def pop_top(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.pop()[1].advance()]


@handles("COPY")
def copy(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.push(state.stack[-instruction.argval]).advance()]


@handles("SWAP")
def swap(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    items, rest = state.pop(instruction.argval)
    return [rest.push(items[-1], *items[1:-1], items[0]).advance()]
