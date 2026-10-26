from collections.abc import Sequence
from dataclasses import replace
from functools import reduce

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import Raise
from udfx.frontend.core.expressions import node_of
from udfx.frontend.core.json import packed
from udfx.frontend.core.logic import choose, disjunction
from udfx.frontend.core.values import Item, Iterator, Json, Value
from udfx.frontend.decode import graph, ops
from udfx.frontend.flow.frame import Flow, Frame
from udfx.frontend.flow.machine import explore, follow, operation_at
from udfx.frontend.operations.iterables import indexed
from udfx.ir import FALSE, TRUE, Comprehension, Element, If, Node, lit
from udfx.ops import Fault, apply
from udfx.types import INT


def target_names(body: Sequence[ops.Op]) -> set[str]:
    names: set[str] = set()
    for op in body:
        if isinstance(op, ops.Store) and not op.cell:
            names.add(op.name)
        elif not (isinstance(op, ops.Unpack) and not op.starred):
            break
    return names


def region_of(frame: Frame, exit: graph.ForEach) -> frozenset[int]:
    return frozenset(block.offset for block in frame.decoded.span(exit.offset, exit.end))


def catches_faults(frame: Frame) -> bool:
    return bool(frame.decoded.catches) or (
        frame.caller is not None and catches_faults(frame.caller[0])
    )


def validate_fault_handlers(frame: Frame) -> None:
    if catches_faults(frame):
        raise Unsupported("comprehension faults with exception handlers")


def comprehension(
    frame: Frame, flow: Flow, exit: graph.ForEach, iterator: Iterator, operation: type[ops.Op]
) -> tuple[Item, list[Fault]]:
    validate_comprehension(frame, exit)
    decoded = frame.decoded
    source, element, item = indexed(iterator, f"{frame.code.co_qualname}:{exit.offset}")
    nested = replace(frame, catching=False, caller=(frame, flow.bindings))
    region = region_of(frame, exit)

    def boundary(current: Flow) -> bool:
        return (
            current.block == flow.block
            or current.block not in region
            or isinstance(operation_at(decoded, current), operation)
        )

    values = {**flow.values, exit.advanced: iterator, exit.item: item}
    start = follow(nested, replace(flow, bindings=flow.bindings.under(TRUE)), exit.body, values)
    cases: list[tuple[Node, Node]] = []
    failures: list[Raise] = []
    dynamic: set[bool] = set()
    for outcome in explore(nested, start, boundary):
        if isinstance(outcome, Raise):
            failures.append(outcome)
            continue
        if not isinstance(outcome, Flow):
            raise Unsupported(exit.reason, frame.where(exit.offset))
        reached = operation_at(decoded, outcome)
        if isinstance(reached, operation) and isinstance(reached, ops.Append | ops.Yield):
            value = outcome.values[reached.value]
            dynamic.add(isinstance(value, Json))
            cases.append(
                (outcome.bindings.path, value.node if isinstance(value, Json) else node_of(value))
            )
        elif outcome.block != flow.block:
            raise Unsupported(exit.reason, frame.where(exit.offset))
    if len(dynamic) > 1:
        raise Unsupported("comprehension mixes dynamic JSON and typed values")
    pending: list[Fault] = []
    if failures and catches_faults(frame):
        if operation is ops.Yield:
            validate_fault_handlers(frame)
        pending.extend(element_faults(source, element, failures, iterator.requires_native))
        failures = []
    node = series(
        frame,
        source,
        element,
        cases,
        [failure.path for failure in failures],
        empty=lit(""),
        requires_native=iterator.requires_native,
    )
    return (Json(packed(node), list) if True in dynamic else Value(node)), pending


def element_faults(
    source: Node, element: Element, failures: list[Raise], requires_native: bool
) -> list[Fault]:
    kinds = list(dict.fromkeys(failure.kind for failure in failures))
    if len(kinds) == 1:
        predicate = reduce(disjunction, (failure.path for failure in failures), FALSE)
        found = Comprehension(source, element, predicate, TRUE, FALSE, requires_native)
        return [Fault(kinds[0], apply("any", found))]
    code = choose(
        [(failure.path, lit(kinds.index(failure.kind) + 1)) for failure in failures]
        + [(TRUE, lit(0))]
    )
    found = Comprehension(
        source, element, code, apply("__ne__", code, lit(0)), FALSE, requires_native
    )
    first = If(apply("bool", found), apply("__getitem__", found, lit(0)), lit(0), INT)
    return [Fault(kind, apply("__eq__", first, lit(index + 1))) for index, kind in enumerate(kinds)]


def validate_comprehension(frame: Frame, exit: graph.ForEach) -> None:
    body = frame.decoded.operations(exit.offset + 1, exit.end)
    if any(isinstance(op, graph.ForEach) or (isinstance(op, ops.Store) and op.cell) for op in body):
        raise Unsupported(exit.reason, frame.where(exit.offset))
    names = target_names(body)
    if not names:
        raise Unsupported(exit.reason, frame.where(exit.offset))
    if any(isinstance(op, ops.Store) and not op.cell and op.name not in names for op in body):
        raise Unsupported("comprehension assignment", frame.where(exit.offset))


def series(
    frame: Frame,
    source: Node,
    element: Element,
    cases: list[tuple[Node, Node]],
    failures: list[Node],
    *,
    empty: Node,
    requires_native: bool = False,
) -> Comprehension:
    node = Comprehension(
        source,
        element,
        choose(cases) if cases else empty,
        reduce(disjunction, (path for path, _ in cases), FALSE),
        reduce(disjunction, failures, FALSE),
        requires_native,
    )
    if node.fault != FALSE:
        validate_fault_handlers(frame)
    return node
