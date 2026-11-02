from __future__ import annotations

from dataclasses import replace

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import Raise, guard
from udfx.frontend.core.expressions import invoke, node_of
from udfx.frontend.core.logic import depends_on
from udfx.frontend.core.values import Item, Iterator, Opaque, Value
from udfx.frontend.decode import graph, ops
from udfx.frontend.flow.collections import collection_loop, is_empty_list
from udfx.frontend.flow.comprehensions import comprehension, region_of, series, target_names
from udfx.frontend.flow.first_iteration import first_iteration
from udfx.frontend.flow.frame import Flow, Frame
from udfx.frontend.flow.machine import Step, divert, exits, explore, follow
from udfx.frontend.operations.iterables import indexed
from udfx.ir import TRUE, Call, Element, Node, lit
from udfx.ops import apply
from udfx.types import FLOAT, INT, STR


def refusal(frame: Frame, exit: graph.ForEach) -> Unsupported:
    return Unsupported(exit.reason, frame.where(exit.offset))


def leave(frame: Frame, flow: Flow, exit: graph.ForEach, values: dict[int, Item]) -> Step:
    return follow(frame, flow, exit.done, {**values, exit.dummy: Opaque(None)})


@exits(graph.ForEach)
def iterate(frame: Frame, flow: Flow, exit: graph.ForEach) -> list[Step]:
    iterator = flow.values[exit.iterator]
    if not isinstance(iterator, Iterator):
        raise refusal(frame, exit)
    below = None if exit.accumulator is None else flow.values[exit.accumulator]
    if exit.accumulator is not None and is_empty_list(below):
        if not iterator.items and iterator.source is None:
            return [leave(frame, flow, exit, dict(flow.values))]
        item, pending = comprehension(frame, flow, exit, iterator, ops.Append)
        path, raised = guard(flow.bindings.path, pending)
        steps = [divert(frame, flow, exit, failure) for failure in raised]
        if path is not None:
            stored = replace(flow, bindings=flow.bindings.under(path))
            steps.append(leave(frame, stored, exit, {**flow.values, exit.accumulator: item}))
        return steps
    if iterator.source is not None:
        first = first_iteration(frame, flow, exit, iterator)
        if first is not None:
            return first
        if not iterator.json and not iterator.requires_native:
            collected = collection_loop(frame, flow, exit, iterator)
            if collected is not None:
                name, item, pending = collected
                path, raised = guard(flow.bindings.path, pending)
                steps = [divert(frame, flow, exit, failure) for failure in raised]
                if path is not None:
                    stored = replace(flow, bindings=flow.bindings.under(path).store(name, item))
                    steps.append(leave(frame, stored, exit, dict(flow.values)))
                return steps
            return reduced_loop(frame, flow, exit, iterator)
        raise refusal(frame, exit)
    if iterator.position == len(iterator.items):
        return [leave(frame, flow, exit, dict(flow.values))]
    values = {
        **flow.values,
        exit.advanced: replace(iterator, position=iterator.position + 1),
        exit.item: iterator.items[iterator.position],
    }
    return [follow(frame, flow, exit.body, values)]


def reduction_target(frame: Frame, flow: Flow, exit: graph.ForEach) -> str:
    decoded = frame.decoded
    body = decoded.operations(exit.offset + 1, exit.end)
    names = target_names(body)
    updates = {op.name for op in body if isinstance(op, ops.Store) and not op.cell} - names
    temporary = updates - flow.bindings.names.keys()
    updates -= temporary
    if len(updates) != 1 or any(
        isinstance(op, graph.ForEach | ops.Delete) or (isinstance(op, ops.Store) and op.cell)
        for op in body
    ):
        raise refusal(frame, exit)
    if any(
        isinstance(op, ops.Local) and not op.cell and not op.clear and op.name in names | temporary
        for op in decoded.operations(exit.end)
    ):
        raise Unsupported("loop locals used after a dynamic loop")
    return next(iter(updates))


def increment(value: Node, accumulator: Element) -> Node:
    terms = []
    while isinstance(value, Call) and value.op.name == "__add__":
        terms.append(value.arguments[1])
        value = value.arguments[0]
        if value == accumulator:
            result = terms.pop()
            for term in reversed(terms):
                result = apply("__add__", result, term)
            return result
        if accumulator.type != STR:
            break
    raise Unsupported("dynamic loops require addition to one accumulator")


def reduced_loop(frame: Frame, flow: Flow, exit: graph.ForEach, iterator: Iterator) -> list[Step]:
    name = reduction_target(frame, flow, exit)
    initial = node_of(flow.bindings.load(name))
    if initial.type not in (STR, INT, FLOAT):
        raise refusal(frame, exit)
    prefix = f"{frame.code.co_qualname}:{exit.offset}"
    accumulator = Element(f"{prefix}:accumulator", initial.type)
    source, element, item = indexed(iterator, f"{prefix}:item")
    nested = replace(frame, catching=False, caller=(frame, flow.bindings))
    region = region_of(frame, exit)
    state = flow.bindings.store(name, Value(accumulator)).under(TRUE)
    values = {**flow.values, exit.advanced: iterator, exit.item: item}
    start = follow(nested, replace(flow, bindings=state), exit.body, values)
    cases: list[tuple[Node, Node]] = []
    failures: list[Node] = []
    for outcome in explore(
        nested, start, lambda current: current.block == flow.block or current.block not in region
    ):
        if isinstance(outcome, Raise):
            failures.append(outcome.path)
            continue
        if not isinstance(outcome, Flow) or outcome.block != flow.block:
            raise refusal(frame, exit)
        value = node_of(outcome.bindings.load(name))
        if value == accumulator:
            continue
        cases.append((outcome.bindings.path, increment(value, accumulator)))
    if any(
        depends_on(node, accumulator)
        for node in (*failures, *(node for case in cases for node in case))
    ):
        raise Unsupported("dynamic loop iterations depend on the accumulator")
    identity = lit("" if initial.type == STR else 0.0 if initial.type == FLOAT else 0)
    items = series(frame, source, element, cases, failures, empty=identity)
    operation = ".join" if initial.type == STR else "sum_float" if initial.type == FLOAT else "sum"
    total = apply(operation, identity, items) if initial.type == STR else apply(operation, items)
    value, pending = invoke("__add__", [initial, total], TypeError)
    path, raised = guard(flow.bindings.path, pending)
    steps = [divert(frame, flow, exit, failure) for failure in raised]
    if path is not None:
        stored = replace(flow, bindings=flow.bindings.under(path).store(name, Value(value)))
        steps.append(leave(frame, stored, exit, dict(flow.values)))
    return steps
