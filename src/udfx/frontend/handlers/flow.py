from __future__ import annotations

import dis

from udfx.errors import Unsupported
from udfx.frontend.calls import node_of, none_test, truth
from udfx.frontend.context import Context, constant
from udfx.frontend.logic import conj, neg
from udfx.frontend.machine import handles
from udfx.frontend.state import Return, State, Step
from udfx.ir import FALSE

JUMP_WHEN_TRUE = frozenset({"POP_JUMP_IF_TRUE", "POP_JUMP_IF_NONE"})


@handles("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE", "POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE")
def branch(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    test, faults = (none_test(item), []) if instruction.opname.endswith("NONE") else truth(item)
    survivor, steps = context.guard(rest, faults)
    if survivor is None:
        return steps
    jump = test if instruction.opname in JUMP_WHEN_TRUE else neg(test)
    for path, index in (
        (conj(survivor.path, neg(jump)), survivor.index + 1),
        (conj(survivor.path, jump), context.target(instruction)),
    ):
        if path != FALSE:
            steps.append(survivor.under(path).goto(index))
    return steps


@handles("JUMP_FORWARD")
def jump_forward(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.goto(context.target(instruction))]


@handles("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT")
def jump_backward(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    raise Unsupported("loops", context.where(state))


@handles("RETURN_VALUE")
def return_value(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item,), _ = state.pop()
    return [Return(state.path, node_of(item))]


@handles("RETURN_CONST")
def return_const(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [Return(state.path, node_of(constant(instruction.argval)))]
