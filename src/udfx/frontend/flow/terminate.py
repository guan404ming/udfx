from __future__ import annotations

from dataclasses import replace

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import Raise, Return
from udfx.frontend.core.exceptions import exception_kind
from udfx.frontend.core.expressions import node_of, returned
from udfx.frontend.core.logic import conjunction, negate
from udfx.frontend.decode import graph
from udfx.frontend.flow.frame import Flow, Frame
from udfx.frontend.flow.machine import Step, divert, exits, follow
from udfx.ir import FALSE


@exits(graph.Jump)
def jump(frame: Frame, flow: Flow, exit: graph.Jump) -> list[Step]:
    return [follow(frame, flow, exit.edge)]


@exits(graph.Branch)
def branch(frame: Frame, flow: Flow, exit: graph.Branch) -> list[Step]:
    test = node_of(flow.values[exit.condition])
    steps: list[Step] = []
    for condition, edge in ((test, exit.then), (negate(test), exit.orelse)):
        path = conjunction(flow.bindings.path, condition)
        if path != FALSE:
            steps.append(follow(frame, replace(flow, bindings=flow.bindings.under(path)), edge))
    return steps


@exits(graph.Return)
def return_(frame: Frame, flow: Flow, exit: graph.Return) -> list[Step]:
    item = flow.values[exit.source]
    return [Return(flow.bindings.path, returned(item, frame.where(exit.offset)))]


@exits(graph.Fail)
def fail(frame: Frame, flow: Flow, exit: graph.Fail) -> list[Step]:
    kind = exception_kind(flow.values[exit.exception])
    return [divert(frame, flow, exit, Raise(flow.bindings.path, kind))]


@exits(graph.Refuse)
def refuse(frame: Frame, flow: Flow, exit: graph.Refuse) -> list[Step]:
    raise Unsupported(exit.reason, frame.where(exit.offset))
