from __future__ import annotations

import dis
from dataclasses import replace
from functools import reduce

from udfx.errors import Unsupported
from udfx.frontend.calls import node_of
from udfx.frontend.constants import constant, constant_value
from udfx.frontend.context import Context
from udfx.frontend.logic import choose, disjunction
from udfx.frontend.machine import emit, explore, handles, unsupported
from udfx.frontend.state import Raise, State, Step
from udfx.frontend.values import MAX_UNROLL, Iterator, Tuple, Value
from udfx.ir import FALSE, TRUE, Comprehension, Element, ListValue, Lit, Node, lit
from udfx.ops import Fault, apply
from udfx.types import STR, Kind, list_of


def source_of(iterator: Iterator) -> Node:
    if iterator.source is not None:
        return iterator.source
    nodes = tuple(node_of(item) for item in iterator.items)
    if nodes and any(node.type != nodes[0].type for node in nodes):
        raise Unsupported("comprehension items must share one type")
    if any(not isinstance(node, Lit) for node in nodes):
        return ListValue(nodes, list_of(nodes[0].type))
    return Lit(
        tuple(constant_value(item) for item in iterator.items),
        list_of(nodes[0].type if nodes else STR),
    )


@handles("GET_ITER")
def get_iter(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    if isinstance(item, Tuple):
        if len(item.items) > MAX_UNROLL:
            raise Unsupported(f"loops and comprehensions exceed {MAX_UNROLL} unrolled items")
        return [rest.push(Iterator(item.items)).advance()]
    if isinstance(item, Value) and not isinstance(item.node, Lit):
        source = node_of(item)
        if source.type.kind is not Kind.LIST:
            raise unsupported(context, state)
        faults = [Fault(TypeError, apply("is_none", source))] if source.type.nullable else []
        return emit(context, rest, Iterator((), source=source), faults)
    try:
        value = constant_value(item)
    except Unsupported:
        raise unsupported(context, state) from None
    container = item.container if isinstance(item, Value) else type(value)
    if container not in (tuple, list) or not isinstance(value, tuple | list):
        raise unsupported(context, state)
    if len(value) > MAX_UNROLL:
        raise Unsupported(f"loops and comprehensions exceed {MAX_UNROLL} unrolled items")
    return [rest.push(Iterator(tuple(constant(member) for member in value))).advance()]


def comprehension(context: Context, state: State, iterator: Iterator, operation: str) -> Node:
    instruction = context.instructions[state.index]
    target = context.target(instruction)
    body = context.instructions[state.index + 1 : target]
    if any(part.opname in {"FOR_ITER", "STORE_DEREF"} for part in body):
        raise unsupported(context, state)
    if not body or body[0].opname != "STORE_FAST":
        raise unsupported(context, state)
    name = body[0].argval
    if any(part.opname == "STORE_FAST" and part.argval != name for part in body):
        raise Unsupported("comprehension assignment", context.where(state))
    source = source_of(iterator)
    if source.type.item is None:
        raise unsupported(context, state)
    element = Element(f"{context.code.co_qualname}:{instruction.offset}:{name}", source.type.item)
    nested = replace(context, protected=())
    start = state.push(Value(element)).under(TRUE).advance()

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
        raise unsupported(context, state)
    if iterator.position == len(iterator.items):
        return [rest.goto(target + 1)]
    item = iterator.items[iterator.position]
    return [rest.push(replace(iterator, position=iterator.position + 1), item).advance()]
