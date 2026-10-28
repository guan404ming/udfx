from dataclasses import replace

from udfx.frontend.core.evaluation import Raise, Return
from udfx.frontend.core.json import dynamic
from udfx.frontend.core.logic import conjunction, negate
from udfx.frontend.core.values import Iterator, Json, Opaque, Value
from udfx.frontend.decode import graph
from udfx.frontend.flow.frame import Flow, Frame
from udfx.frontend.flow.machine import Step, explore, follow
from udfx.frontend.operations.iterables import source_of
from udfx.ir import FALSE, Call, lit
from udfx.ops import apply
from udfx.semantics.assumptions import known, predicates


def repeats(frame: Frame, header: int, body: int) -> bool:
    pending = [body]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current == header:
            return True
        if current in seen:
            continue
        seen.add(current)
        match frame.decoded.blocks[current].exit:
            case graph.Jump(edge=edge):
                pending.append(edge.target)
            case graph.Branch(then=then, orelse=orelse):
                pending.extend((then.target, orelse.target))
            case graph.ForEach(body=repeat, done=done):
                pending.extend((repeat.target, done.target))
            case _:
                pass
    return False


def first_iteration(
    frame: Frame, flow: Flow, exit: graph.ForEach, iterator: Iterator
) -> list[Step] | None:
    if iterator.value is not None or repeats(frame, flow.block, exit.body.target):
        return None
    source = source_of(iterator)
    document = None
    if iterator.json and isinstance(source, Call) and source.op.name == "json.tokens":
        raw = source.arguments[0]
        source = apply("json.items", dynamic(raw))
        if known(
            apply("json.is_type", raw, lit("list")), dict(predicates(flow.bindings.path, True))
        ):
            document = dynamic(raw)
    length = apply("len", source) if document is None else apply("json.length", document)
    present = apply("__gt__", length, lit(0))
    path = conjunction(flow.bindings.path, present)
    value = (
        apply("__getitem__", source, lit(0))
        if document is None
        else apply("json.item", document, lit(0))
    )
    item = Json(value) if iterator.json else Value(value)
    values = {**flow.values, exit.advanced: iterator, exit.item: item}
    start = follow(frame, replace(flow, bindings=flow.bindings.under(path)), exit.body, values)
    outcomes = explore(frame, start, lambda current: current.block == flow.block)
    if any(not isinstance(outcome, Return | Raise) for outcome in outcomes):
        return None
    empty = conjunction(flow.bindings.path, negate(present))
    if empty != FALSE:
        state = replace(flow, bindings=flow.bindings.under(empty))
        outcomes.append(follow(frame, state, exit.done, {**flow.values, exit.dummy: Opaque(None)}))
    return outcomes
