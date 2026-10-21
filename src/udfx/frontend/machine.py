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

FEATURES: dict[str, str] = {
    "GET_ITER": "loops and comprehensions",
    "FOR_ITER": "loops and comprehensions",
    "JUMP_BACKWARD": "loops and comprehensions",
    "JUMP_BACKWARD_NO_INTERRUPT": "loops and comprehensions",
    "BUILD_LIST": "list and set literals",
    "BUILD_SET": "list and set literals",
    "LIST_APPEND": "list and set literals",
    "LIST_EXTEND": "list and set literals",
    "STORE_GLOBAL": "global assignment",
    "DELETE_GLOBAL": "global assignment",
    "IMPORT_NAME": "imports",
    "BEFORE_WITH": "with blocks",
    "LOAD_SPECIAL": "with blocks",
    "WITH_EXCEPT_START": "with blocks",
    "RAISE_VARARGS": "raise statements",
    "YIELD_VALUE": "generators",
    "RETURN_GENERATOR": "generators",
    "CALL_FUNCTION_EX": "unpacked call arguments",
    "DICT_MERGE": "unpacked call arguments",
}


def handles(*opnames: str) -> Callable[[Handler], Handler]:
    def register(handler: Handler) -> Handler:
        for opname in opnames:
            if opname in HANDLERS:
                raise ValueError(f"duplicate handler {opname}")
            HANDLERS[opname] = handler
        return handler

    return register


def emit(context: Context, state: State, result: Value, faults: Sequence[Fault]) -> list[Step]:
    survivor, steps = context.guard(state, faults)
    if survivor is not None:
        steps.append(survivor.push(result).advance())
    return steps


def unsupported(context: Context, state: State) -> Unsupported:
    opname = context.instructions[state.index].opname
    return Unsupported(FEATURES.get(opname, f"opcode {opname}"), context.where(state))


def explore(context: Context, start: State) -> list[Outcome]:
    pending: list[State] = [start]
    outcomes: list[Outcome] = []
    steps = 0
    while pending:
        steps += 1
        if steps > MAX_STEPS:
            raise Unsupported(
                f"more than {MAX_STEPS} execution paths; "
                "the function has too many independent branches for one expression"
            )
        state = pending.pop()
        instruction = context.instructions[state.index]
        handler = HANDLERS.get(instruction.opname)
        if handler is None:
            raise unsupported(context, state)
        produced = handler(context, state, instruction)
        outcomes.extend(step for step in produced if not isinstance(step, State))
        pending.extend(reversed([step for step in produced if isinstance(step, State)]))
    return outcomes
