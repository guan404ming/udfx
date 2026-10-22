from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import Outcome, Raise
from udfx.frontend.core.values import NULL as NULL_ITEM
from udfx.frontend.core.values import Item, Opaque
from udfx.frontend.decode import ops
from udfx.frontend.decode.graph import Decoded
from udfx.frontend.flow.frame import Flow, Frame, Result

MAX_STEPS = 10_000

Step = Flow | Outcome
Evaluator = Callable[[Frame, Flow, Any], list[Result]]
Exit = Callable[[Frame, Flow, Any], list[Step]]

EVALUATORS: dict[type, Evaluator] = {}
EXITS: dict[type, Exit] = {}


def evaluates(*kinds: type) -> Callable[[Evaluator], Evaluator]:
    def register(evaluator: Evaluator) -> Evaluator:
        for kind in kinds:
            EVALUATORS[kind] = evaluator
        return evaluator

    return register


def exits(*kinds: type) -> Callable[[Exit], Exit]:
    def register(handler: Exit) -> Exit:
        for kind in kinds:
            EXITS[kind] = handler
        return handler

    return register


def operation_at(decoded: Decoded, flow: Flow) -> ops.Op:
    block = decoded.blocks[flow.block]
    return block.ops[flow.position] if flow.position < len(block.ops) else block.exit


def outputs(op: ops.Op) -> tuple[int, ...]:
    results = getattr(op, "results", None)
    if results is not None:
        return results
    result = getattr(op, "result", None)
    return () if result is None else (result,)


def read(values: Mapping[int, Item], identifier: int) -> Item:
    return NULL_ITEM if identifier == ops.NULL else values[identifier]


def optional(values: Mapping[int, Item], identifier: int | None) -> Item | None:
    return None if identifier is None else values[identifier]


def follow(frame: Frame, flow: Flow, edge: ops.Edge, values: dict[int, Item] | None = None) -> Flow:
    current = flow.values if values is None else values
    parameters = frame.decoded.blocks[edge.target].parameters
    bound = {
        parameter: read(current, argument)
        for parameter, argument in zip(parameters, edge.arguments, strict=True)
    }
    return Flow(edge.target, 0, flow.bindings, {**current, **bound})


def divert(frame: Frame, flow: Flow, op: ops.Op, failure: Raise) -> Step:
    handler = op.handler
    if handler is None or not frame.catching:
        return failure
    return follow(
        frame,
        replace(flow, bindings=flow.bindings.under(failure.path)),
        handler.edge,
        {**flow.values, handler.exception: Opaque(failure.kind)},
    )


def settle(frame: Frame, flow: Flow, op: ops.Op, results: Sequence[Result]) -> list[Step]:
    steps: list[Step] = []
    for result in results:
        if isinstance(result, Raise):
            steps.append(divert(frame, flow, op, result))
            continue
        bound = dict(zip(outputs(op), result.items, strict=True))
        steps.append(Flow(flow.block, flow.position + 1, result.bindings, {**flow.values, **bound}))
    return steps


def step(frame: Frame, flow: Flow) -> list[Step]:
    op = operation_at(frame.decoded, flow)
    if flow.position < len(frame.decoded.blocks[flow.block].ops):
        return settle(frame, flow, op, EVALUATORS[type(op)](frame, flow, op))
    return EXITS[type(op)](frame, flow, op)


def explore(frame: Frame, start: Flow, stop: Callable[[Flow], bool] | None = None) -> list[Step]:
    pending: list[Flow] = [start]
    outcomes: list[Step] = []
    count = 0
    while pending:
        flow = pending.pop()
        count += operation_at(frame.decoded, flow).weight
        if count > MAX_STEPS:
            raise Unsupported(
                f"more than {MAX_STEPS} execution paths; "
                "the function has too many independent branches for one expression"
            )
        if stop is not None and stop(flow):
            outcomes.append(flow)
            continue
        produced = step(frame, flow)
        outcomes.extend(item for item in produced if not isinstance(item, Flow))
        pending.extend(reversed([item for item in produced if isinstance(item, Flow)]))
    return outcomes
