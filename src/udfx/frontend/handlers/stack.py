from __future__ import annotations

import dis

from udfx.errors import Unsupported
from udfx.frontend.constants import constant, literal, scalar_map
from udfx.frontend.context import Context
from udfx.frontend.machine import handles
from udfx.frontend.state import State, Step
from udfx.frontend.values import Opaque


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


@handles("BUILD_CONST_KEY_MAP")
def build_const_key_map(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    items, rest = state.pop(instruction.argval + 1)
    *values, keys = items
    names = literal(keys)
    if not isinstance(names, tuple):
        raise Unsupported(f"dict keys {names!r}", context.where(state))
    entries = scalar_map(zip(names, map(literal, values), strict=True))
    return [rest.push(Opaque(entries)).advance()]


@handles("BUILD_MAP")
def build_map(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    items, rest = state.pop(2 * instruction.argval)
    constants = [literal(item) for item in items]
    entries = scalar_map(zip(constants[::2], constants[1::2], strict=True))
    return [rest.push(Opaque(entries)).advance()]
