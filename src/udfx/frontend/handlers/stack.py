from __future__ import annotations

import dis

from udfx.errors import Unsupported
from udfx.frontend.calls import node_of
from udfx.frontend.constants import constant, constant_value, scalar_map
from udfx.frontend.context import Context
from udfx.frontend.machine import handles
from udfx.frontend.state import State, Step
from udfx.frontend.values import NULL, Opaque, Set, Tuple, Value
from udfx.ir import ListValue, Lit
from udfx.types import STR, list_of


@handles("RESUME", "NOP", "CACHE")
def skip(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.advance()]


@handles("LOAD_FAST", "LOAD_FAST_CHECK")
def load_fast(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.push(state.load(instruction.argval)).advance()]


@handles("STORE_FAST")
def store_fast(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    return [rest.store(instruction.argval, item).advance()]


@handles("LOAD_FAST_AND_CLEAR")
def load_fast_and_clear(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [
        state.push(state.names.get(instruction.argval, NULL)).delete(instruction.argval).advance()
    ]


@handles("LOAD_CONST")
def load_const(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.push(constant(instruction.argval)).advance()]


@handles("POP_TOP")
def pop_top(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.pop()[1].advance()]


@handles("COPY")
def copy(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.push(state.stack[-instruction.argval]).advance()]


@handles("SWAP")
def swap(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    items, rest = state.pop(instruction.argval)
    return [rest.push(items[-1], *items[1:-1], items[0]).advance()]


@handles("BUILD_CONST_KEY_MAP")
def build_const_key_map(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    items, rest = state.pop(instruction.argval + 1)
    *values, keys = items
    names = constant_value(keys)
    if not isinstance(names, tuple):
        raise Unsupported(f"dict keys {names!r}", context.where(state))
    entries = scalar_map(zip(names, map(constant_value, values), strict=True))
    return [rest.push(Opaque(entries)).advance()]


@handles("BUILD_MAP")
def build_map(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    items, rest = state.pop(2 * instruction.argval)
    constants = [constant_value(item) for item in items]
    entries = scalar_map(zip(constants[::2], constants[1::2], strict=True))
    return [rest.push(Opaque(entries)).advance()]


@handles("DELETE_FAST")
def delete_fast(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.delete(instruction.argval).advance()]


@handles("BUILD_TUPLE")
def build_tuple(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    items, rest = state.pop(instruction.argval)
    try:
        values = tuple(constant_value(item) for item in items)
    except Unsupported:
        return [rest.push(Tuple(items)).advance()]
    return [rest.push(constant(values)).advance()]


@handles("BUILD_LIST", "BUILD_SET")
def build_container(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    items, rest = state.pop(instruction.argval)
    if instruction.opname == "BUILD_SET":
        try:
            return [rest.push(constant({constant_value(item) for item in items})).advance()]
        except Unsupported:
            nodes = tuple(node_of(item) for item in items)
            if any(node.type != STR for node in nodes):
                raise Unsupported("set requires string items", context.where(state)) from None
            return [rest.push(Set(ListValue(nodes, list_of(STR)))).advance()]
    nodes = tuple(node_of(item) for item in items)
    if nodes and any(node.type != nodes[0].type for node in nodes):
        raise Unsupported("list items must share one type", context.where(state))
    result = list_of(nodes[0].type if nodes else STR)
    value = (
        Lit(tuple(node.value for node in nodes if isinstance(node, Lit)), result)
        if all(isinstance(node, Lit) for node in nodes)
        else ListValue(nodes, result)
    )
    return [rest.push(Value(value, list)).advance()]


@handles("LIST_EXTEND", "SET_UPDATE")
def extend_container(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    members = constant_value(item)
    items, base = rest.pop(instruction.argval)
    current = constant_value(items[0])
    if not isinstance(members, tuple | list | set | frozenset) or not isinstance(
        current, tuple | list | set | frozenset
    ):
        raise Unsupported("list and set literals", context.where(state))
    values = [*current, *members]
    value = set(values) if instruction.opname == "SET_UPDATE" else values
    return [base.push(constant(value), *items[1:]).advance()]
