from __future__ import annotations

import dis

from udfx.errors import Unsupported
from udfx.frontend.calls import call
from udfx.frontend.context import Context
from udfx.frontend.machine import emit, handles
from udfx.frontend.state import State, Step
from udfx.frontend.values import NULL, Method, Null, Value


@handles("PUSH_NULL")
def push_null(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.push(NULL).advance()]


@handles("LOAD_GLOBAL")
def load_global(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    item = context.global_(instruction.argval)
    pushed = (
        state.push(NULL, item)
        if instruction.arg is not None and instruction.arg & 1
        else state.push(item)
    )
    return [pushed.advance()]


@handles("LOAD_ATTR")
def load_attr(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    if instruction.arg is None or not instruction.arg & 1:
        raise Unsupported(f"attribute load {instruction.argval}", context.where(state))
    (owner,), rest = state.pop()
    return [rest.push(Method(instruction.argval), owner).advance()]


@handles("CALL")
def call_(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    count = instruction.arg or 0
    items, rest = state.pop(count + 2)
    head, second, *others = items
    callee, arguments = (second, others) if isinstance(head, Null) else (head, [second, *others])
    node, faults = call(callee, arguments)
    return emit(context, rest, Value(node), faults)
