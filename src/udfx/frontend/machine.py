from __future__ import annotations

import dis
from collections.abc import Callable, Sequence

from udfx.errors import Unsupported
from udfx.frontend.context import Context
from udfx.frontend.state import Outcome, State, Step
from udfx.frontend.values import Value
from udfx.ops import Fault

Handler = Callable[[Context, State, dis.Instruction], list[Step]]

HANDLERS: dict[str, Handler] = {}

MAX_STEPS = 10_000


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


def explore(ctx: Context, start: State) -> list[Outcome]:
    pending: list[State] = [start]
    outcomes: list[Outcome] = []
    steps = 0
    while pending:
        steps += 1
        if steps > MAX_STEPS:
            raise Unsupported("path explosion")
        state = pending.pop()
        inst = ctx.instructions[state.index]
        handler = HANDLERS.get(inst.opname)
        if handler is None:
            raise Unsupported(f"opcode {inst.opname}", ctx.where(state))
        for step in reversed(handler(ctx, state, inst)):
            if isinstance(step, State):
                pending.append(step)
            else:
                outcomes.append(step)
    return outcomes
