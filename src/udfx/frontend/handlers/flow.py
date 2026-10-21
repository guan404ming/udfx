from __future__ import annotations

import dis

from udfx.errors import Unsupported
from udfx.frontend.calls import node_of, none_test, truth
from udfx.frontend.constants import constant
from udfx.frontend.context import Context
from udfx.frontend.logic import conjunction, negate
from udfx.frontend.machine import handles, unsupported
from udfx.frontend.state import Return, State, Step
from udfx.frontend.values import Iterator, Opaque, Tuple, Value
from udfx.ir import FALSE

JUMP_WHEN_TRUE = frozenset({"POP_JUMP_IF_TRUE", "POP_JUMP_IF_NONE"})


@handles("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE", "POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE")
def branch(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    test, faults = (none_test(item), []) if instruction.opname.endswith("NONE") else truth(item)
    survivor, steps = context.guard(rest, faults)
    if survivor is None:
        return steps
    jump = test if instruction.opname in JUMP_WHEN_TRUE else negate(test)
    for path, index in (
        (conjunction(survivor.path, negate(jump)), survivor.index + 1),
        (conjunction(survivor.path, jump), context.target(instruction)),
    ):
        if path != FALSE:
            steps.append(survivor.under(path).goto(index))
    return steps


@handles("JUMP_FORWARD")
def jump_forward(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.goto(context.target(instruction))]


@handles("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT")
def jump_backward(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    target = context.target(instruction)
    if context.instructions[target].opname == "FOR_ITER" and any(
        isinstance(item, Iterator) for item in state.stack
    ):
        return [state.goto(target)]
    raise unsupported(context, state)


@handles("RETURN_VALUE", "RETURN_CONST")
def return_value(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    item = constant(instruction.argval) if instruction.opname == "RETURN_CONST" else state.stack[-1]
    if (
        isinstance(item, Tuple)
        or (isinstance(item, Value) and item.container is tuple)
        or (isinstance(item, Opaque) and isinstance(item.value, tuple))
    ):
        raise Unsupported(
            "tuple returns are unsupported; return a scalar or list", context.where(state)
        )
    return [Return(state.path, node_of(item))]
