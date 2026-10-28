from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import reduce
from types import FunctionType
from typing import cast

from udfx.errors import Unsupported
from udfx.frontend.core.bindings import Bindings
from udfx.frontend.core.evaluation import Raise, Return, single
from udfx.frontend.core.expressions import node_of, truth
from udfx.frontend.core.logic import choose, conjunction, disjunction, negate
from udfx.frontend.core.merge import merge
from udfx.frontend.core.tracing import FOLDED
from udfx.frontend.core.values import Function, Generator, Item, Null, Opaque, Value, wrapped
from udfx.frontend.decode import ops
from udfx.frontend.flow.comprehensions import comprehension, validate_fault_handlers
from udfx.frontend.flow.frame import Flow, Frame, Produced, Result
from udfx.frontend.flow.machine import evaluates, explore, optional
from udfx.frontend.operations.builtins import call
from udfx.frontend.operations.iterables import indexed, iterator_of, source_of
from udfx.frontend.operations.library import (
    ordinary_call,
    reject_rows,
    require_empty_keywords,
    unpacked_arguments,
)
from udfx.frontend.resolve.constants import constant, scalar_map
from udfx.frontend.resolve.frozen import closure
from udfx.frontend.resolve.functions import bind_helper, helper_signature
from udfx.ir import FALSE, TRUE, Comprehension, Lit, Node, lit

MAX_DEPTH = 32


def dispatch(
    frame: Frame,
    op: ops.Op,
    bindings: Bindings,
    callee: Item,
    arguments: Sequence[Item],
    *,
    keywords: Mapping[str, Item],
) -> list[Result]:
    if isinstance(callee, Function):
        return inline(frame, bindings, callee, arguments, keywords)
    reject_rows(arguments, keywords, frame.where(op.offset))
    mapping = isinstance(callee, Opaque) and (callee.value is map or callee.value is filter)
    if mapping and not keywords:
        return transformed(frame, op, bindings, cast("Opaque", callee).value is filter, arguments)
    return frame.settle(bindings, ordinary_call(callee, arguments, keywords, bindings.path))


def inline(
    frame: Frame,
    bindings: Bindings,
    function: Function,
    arguments: Sequence[Item],
    keywords: Mapping[str, Item],
) -> list[Result]:
    code = function.code
    if frame.depth >= MAX_DEPTH:
        raise Unsupported(f"recursive helper {code.co_qualname}")
    callee, defaults, keyword_defaults, free = enter(frame, bindings, function)
    if code.co_name == "<genexpr>" and len(arguments) == 1:
        return generator(callee, bindings, arguments[0], free)
    for item in (*defaults, *keyword_defaults.values()):
        if isinstance(item, Opaque) and isinstance(item.value, dict):
            scalar_map(item.value.items())
        elif not isinstance(item, Value) or not isinstance(item.node, Lit):
            raise Unsupported("helper with non-constant parameter defaults")
    signature = helper_signature(code, defaults, keyword_defaults)
    names = {**bind_helper(signature, arguments, keywords), **free}
    joining = FOLDED.get() is not None
    start = Flow(callee.decoded.entry, 0, Bindings(names, TRUE if joining else bindings.path), {})
    outcomes = explore(callee, start)
    if joining:
        returns = [outcome for outcome in outcomes if isinstance(outcome, Return)]
        failures = [outcome for outcome in outcomes if isinstance(outcome, Raise)]
        results: list[Result] = [
            Raise(conjunction(bindings.path, outcome.path), outcome.kind) for outcome in failures
        ]
        if returns:
            faults = [outcome.path for outcome in failures]
            path = conjunction(bindings.path, negate(reduce(disjunction, faults, FALSE)))
            results.append(Produced(bindings.under(path), (wrapped(merge(returns, faults)),)))
        return results
    results = []
    for outcome in outcomes:
        if isinstance(outcome, Return):
            results.append(Produced(bindings.under(outcome.path), (wrapped(outcome.value),)))
        elif isinstance(outcome, Raise):
            results.append(outcome)
    return results


def enter(
    frame: Frame, bindings: Bindings, function: Function
) -> tuple[Frame, tuple[Item, ...], Mapping[str, Item], Mapping[str, Item]]:
    code = function.code
    caller = (frame, bindings)
    if isinstance(function.home, FunctionType):
        helper = function.home
        defaults = tuple(constant(value) for value in helper.__defaults__ or ())
        keyword_defaults = {
            name: constant(value) for name, value in (helper.__kwdefaults__ or {}).items()
        }
        callee = Frame.from_code(
            code, helper.__globals__, helper.__builtins__, caller, frame.snapshot
        )
        return callee, defaults, keyword_defaults, closure(helper)
    home = cast("Frame", function.home)
    outer = frame.locate(bindings, home)
    free = {
        name: outer.load(cell) for name, cell in zip(code.co_freevars, function.cells, strict=True)
    }
    callee = Frame.from_code(code, home.scope, home.builtins, caller, frame.snapshot)
    return callee, function.defaults, function.keyword_defaults, free


def generator(
    callee: Frame, bindings: Bindings, iterable: Item, free: Mapping[str, Item]
) -> list[Result]:
    iterator, pending = iterator_of(iterable)
    block, exit = callee.decoded.loop()
    names = {".0": iterator, **free}
    values: dict[int, Item] = dict.fromkeys(block.parameters, Opaque(None))
    values[exit.iterator] = iterator
    start = Flow(block.offset, 0, Bindings(names, bindings.path), values)
    if iterator.items or iterator.source is not None:
        item, faults = comprehension(callee, start, exit, iterator, ops.Yield)
        node = node_of(item)
        pending.extend(faults)
    else:
        node = source_of(iterator)
    return callee.settle(bindings, single(Generator(node), pending))


def transformed(
    frame: Frame, op: ops.Op, bindings: Bindings, filtering: bool, arguments: Sequence[Item]
) -> list[Result]:
    if len(arguments) != 2:
        raise Unsupported("map and filter require a function and one iterable")
    function, iterable = arguments
    iterator, pending = iterator_of(iterable)
    if not iterator.items and iterator.source is None:
        return frame.settle(bindings, single(Generator(source_of(iterator)), pending))
    source, element, item = indexed(iterator, f"{frame.code.co_qualname}:{op.offset}:map")
    cases: list[tuple[Node, Node]] = []
    failures: list[Node] = []
    for outcome in transform_item(frame, bindings, function, item, filtering):
        if isinstance(outcome, Raise):
            failures.append(outcome.path)
            continue
        result = outcome.items[-1]
        path = outcome.bindings.path
        if filtering:
            value, faults = truth(result)
            failures.extend(conjunction(path, fault.when) for fault in faults)
        else:
            value = node_of(result)
        cases.append((path, value))
    value = choose(cases) if cases else lit("")
    fault = reduce(disjunction, failures, FALSE)
    node = Comprehension(
        source,
        element,
        node_of(item) if filtering else value,
        (value if cases else FALSE) if filtering else TRUE,
        fault,
        iterator.requires_native,
    )
    if fault != FALSE:
        validate_fault_handlers(frame)
    return frame.settle(bindings, single(Generator(node), pending))


def transform_item(
    frame: Frame, bindings: Bindings, function: Item, item: Item, filtering: bool
) -> list[Result]:
    nested = replace(frame, catching=False, caller=(frame, bindings))
    start = bindings.under(TRUE)
    if isinstance(function, Function):
        return inline(nested, start, function, [item], {})
    value, faults = (
        truth(item)
        if filtering and isinstance(function, Value) and function.node == lit(None)
        else call(function, [item])
    )
    return nested.settle(start, single(Value(value), faults))


def callee_of(
    values: Mapping[int, Item], op: ops.Call | ops.UnpackedCall
) -> tuple[Item, Item | None]:
    callee = values[op.callee]
    receiver = optional(values, op.receiver)
    return callee, None if isinstance(receiver, Null) else receiver


@evaluates(ops.Call)
def call_function(frame: Frame, flow: Flow, op: ops.Call) -> list[Result]:
    callee, receiver = callee_of(flow.values, op)
    arguments = [flow.values[identifier] for identifier in op.arguments]
    if receiver is not None:
        arguments.insert(0, receiver)
    keywords = {name: flow.values[identifier] for name, identifier in op.keywords}
    return dispatch(frame, op, flow.bindings, callee, arguments, keywords=keywords)


@evaluates(ops.UnpackedCall)
def call_unpacked(frame: Frame, flow: Flow, op: ops.UnpackedCall) -> list[Result]:
    values, where = flow.values, frame.where(op.offset)
    if op.keywords is not None:
        require_empty_keywords(values[op.keywords], where)
    callee, receiver = callee_of(values, op)
    if receiver is not None:
        raise Unsupported("unpacked call arguments", where)
    items = unpacked_arguments(values[op.arguments], where)
    return dispatch(frame, op, flow.bindings, callee, items, keywords={})
