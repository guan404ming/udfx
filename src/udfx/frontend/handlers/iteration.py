from __future__ import annotations

import dis
from dataclasses import replace
from functools import reduce

from udfx.errors import Unsupported
from udfx.frontend.calls import invoke, node_of
from udfx.frontend.context import Context
from udfx.frontend.iterables import indexed, iterator_of
from udfx.frontend.logic import choose, depends_on, disjunction
from udfx.frontend.machine import emit, explore, handles, unsupported
from udfx.frontend.state import Raise, State, Step
from udfx.frontend.values import Iterator, Value
from udfx.ir import FALSE, TRUE, Call, Comprehension, Element, Lit, Node, lit
from udfx.ops import apply
from udfx.types import FLOAT, INT, STR


@handles("GET_ITER")
def get_iter(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    iterator, faults = iterator_of(item)
    return emit(context, rest, iterator, faults)


def comprehension(context: Context, state: State, iterator: Iterator, operation: str) -> Node:
    instruction = context.instructions[state.index]
    target = context.target(instruction)
    body = context.instructions[state.index + 1 : target]
    if any(part.opname in {"FOR_ITER", "STORE_DEREF"} for part in body):
        raise unsupported(context, state)
    names = target_names(body)
    if not names:
        raise unsupported(context, state)
    if any(part.opname == "STORE_FAST" and part.argval not in names for part in body):
        raise Unsupported("comprehension assignment", context.where(state))
    source, element, item = indexed(iterator, f"{context.code.co_qualname}:{instruction.offset}")
    nested = replace(context, protected=())
    start = state.push(item).under(TRUE).advance()

    def boundary(current: State) -> bool:
        return (
            current.index <= state.index
            or current.index >= target
            or context.instructions[current.index].opname == operation
        )

    cases: list[tuple[Node, Node]] = []
    faults: list[Node] = []
    for outcome in explore(nested, start, boundary):
        if isinstance(outcome, Raise):
            faults.append(outcome.path)
        elif isinstance(outcome, State) and context.instructions[outcome.index].opname == operation:
            cases.append((outcome.path, node_of(outcome.stack[-1])))
        elif not isinstance(outcome, State) or outcome.index != state.index:
            raise unsupported(context, state)
    node = Comprehension(
        source,
        element,
        choose(cases) if cases else lit(""),
        reduce(disjunction, (path for path, _ in cases), FALSE),
        reduce(disjunction, faults, FALSE),
        iterator.requires_native,
    )
    if node.fault != FALSE:
        validate_fault_handlers(context)
    return node


def validate_fault_handlers(context: Context) -> None:
    while True:
        if any(instruction.opname == "PUSH_EXC_INFO" for instruction in context.instructions):
            raise Unsupported("comprehension faults with exception handlers")
        if context.caller is None:
            return
        context = context.caller[0]


@handles("FOR_ITER")
def for_iter(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (iterator,), rest = state.pop()
    if not isinstance(iterator, Iterator):
        raise unsupported(context, state)
    target = context.target(instruction)
    if (
        rest.stack
        and isinstance(rest.stack[-1], Value)
        and rest.stack[-1].container is list
        and isinstance(rest.stack[-1].node, Lit)
        and rest.stack[-1].node.value == ()
    ):
        if not iterator.items and iterator.source is None:
            return [rest.goto(target + 1)]
        node = comprehension(context, state, iterator, "LIST_APPEND")
        return [rest.pop()[1].push(Value(node)).goto(target + 1)]
    if iterator.source is not None:
        if iterator.value is not None and not iterator.requires_native:
            return reduced_loop(context, state, iterator)
        raise unsupported(context, state)
    if iterator.position == len(iterator.items):
        return [rest.goto(target + 1)]
    item = iterator.items[iterator.position]
    return [rest.push(replace(iterator, position=iterator.position + 1), item).advance()]


def target_names(body: tuple[dis.Instruction, ...]) -> set[str]:
    names: set[str] = set()
    for part in body:
        if part.opname == "STORE_FAST":
            names.add(part.argval)
        elif part.opname != "UNPACK_SEQUENCE":
            break
    return names


def reduced_loop(context: Context, state: State, iterator: Iterator) -> list[Step]:
    target = context.target(context.instructions[state.index])
    body = context.instructions[state.index + 1 : target]
    names = target_names(body)
    updates = {part.argval for part in body if part.opname == "STORE_FAST"} - names
    if len(updates) != 1 or any(
        part.opname in {"FOR_ITER", "STORE_DEREF", "DELETE_FAST"} for part in body
    ):
        raise unsupported(context, state)
    if any(
        part.opname in {"LOAD_FAST", "LOAD_FAST_CHECK"} and part.argval in names
        for part in context.instructions[target:]
    ):
        raise Unsupported("loop targets used after a dynamic loop")
    name = next(iter(updates))
    initial = node_of(state.load(name))
    if initial.type not in (STR, INT, FLOAT):
        raise unsupported(context, state)
    accumulator = Element(f"{context.code.co_qualname}:{state.index}:accumulator", initial.type)
    source, element, item = indexed(iterator, f"{context.code.co_qualname}:{state.index}:item")
    nested = replace(context, protected=(), caller=(context, state))
    start = state.store(name, Value(accumulator)).push(item).under(TRUE).advance()
    cases: list[tuple[Node, Node]] = []
    failures: list[Node] = []
    for outcome in explore(
        nested, start, lambda current: current.index <= state.index or current.index >= target
    ):
        if isinstance(outcome, Raise):
            failures.append(outcome.path)
            continue
        if not isinstance(outcome, State) or outcome.index != state.index:
            raise unsupported(context, state)
        value = node_of(outcome.load(name))
        if value == accumulator:
            continue
        if (
            not isinstance(value, Call)
            or value.op.name != "__add__"
            or value.arguments[0] != accumulator
        ):
            raise Unsupported("dynamic loops require addition to one accumulator")
        cases.append((outcome.path, value.arguments[1]))
    if any(
        depends_on(node, accumulator)
        for node in (*failures, *(node for case in cases for node in case))
    ):
        raise Unsupported("dynamic loop iterations depend on the accumulator")
    identity = lit("" if initial.type == STR else 0.0 if initial.type == FLOAT else 0)
    fault = reduce(disjunction, failures, FALSE)
    values = Comprehension(
        source,
        element,
        choose(cases) if cases else identity,
        reduce(disjunction, (path for path, _value in cases), FALSE),
        fault,
    )
    if fault != FALSE:
        validate_fault_handlers(context)
    operation = ".join" if initial.type == STR else "sum_float" if initial.type == FLOAT else "sum"
    total = apply(operation, identity, values) if initial.type == STR else apply(operation, values)
    value, faults = invoke("__add__", [initial, total], TypeError)
    survivor, steps = context.guard(state.pop()[1], faults)
    if survivor is not None:
        steps.append(survivor.store(name, Value(value)).goto(target + 1))
    return steps
