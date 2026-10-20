from __future__ import annotations

import dis

from udfx.errors import Unsupported
from udfx.frontend.calls import node_of, none_test, truth
from udfx.frontend.context import Context
from udfx.frontend.logic import conj, neg
from udfx.frontend.machine import handles
from udfx.frontend.state import Return, State, Step
from udfx.ir import FALSE

JUMP_WHEN_TRUE = frozenset({"POP_JUMP_IF_TRUE", "POP_JUMP_IF_NONE"})


@handles("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE", "POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE")
def branch(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    test, faults = (none_test(item), []) if inst.opname.endswith("NONE") else truth(item)
    survivor, steps = ctx.guard(rest, faults)
    if survivor is None:
        return steps
    jump = test if inst.opname in JUMP_WHEN_TRUE else neg(test)
    for path, index in (
        (conj(survivor.path, neg(jump)), survivor.index + 1),
        (conj(survivor.path, jump), ctx.target(inst)),
    ):
        if path != FALSE:
            steps.append(survivor.under(path).goto(index))
    return steps


@handles("JUMP_FORWARD")
def jump_forward(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    return [state.goto(ctx.target(inst))]


@handles("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT")
def jump_backward(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    raise Unsupported("loops", ctx.where(state))


@handles("RETURN_VALUE")
def return_value(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    (item,), _ = state.pop()
    return [Return(state.path, node_of(item))]


@handles("RETURN_CONST")
def return_const(ctx: Context, state: State, inst: dis.Instruction) -> list[Step]:
    return [Return(state.path, node_of(ctx.constant(inst.argval)))]
