from __future__ import annotations

import dis

from udfx.errors import Unsupported
from udfx.frontend.constants import constant, constant_value, scalar_map
from udfx.frontend.context import Context
from udfx.frontend.invocation import invoke, node_of
from udfx.frontend.machine import handles
from udfx.frontend.state import State, Step
from udfx.frontend.values import NULL, Opaque, Set, Tuple, Value
from udfx.ir import ListValue, Lit, lit
from udfx.ops import Fault, apply
from udfx.types import INT, STR, Kind, list_of


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
    entries = dict(zip(names, map(constant_value, values), strict=True))
    return [rest.push(Opaque(entries)).advance()]


@handles("BUILD_MAP")
def build_map(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    items, rest = state.pop(2 * instruction.argval)
    constants = [constant_value(item) for item in items]
    scalar_map((key, None) for key in constants[::2])
    entries = dict(zip(constants[::2], constants[1::2], strict=True))
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
            if any(
                node.type != nodes[0].type
                or node.type.kind not in {Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL}
                for node in nodes
            ):
                raise Unsupported(
                    "set requires matching scalar items", context.where(state)
                ) from None
            return [rest.push(Set(ListValue(nodes, list_of(nodes[0].type)))).advance()]
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


@handles("UNPACK_SEQUENCE", "UNPACK_EX")
def unpack_sequence(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    starred = instruction.opname == "UNPACK_EX"
    before = instruction.argval & 255 if starred else instruction.argval
    after = instruction.argval >> 8 if starred else 0
    match item:
        case Tuple(items=items):
            pass
        case Value(node=Lit(value=tuple() as values), container=container) if container is tuple:
            items = tuple(constant(value) for value in values)
        case Opaque(value=tuple() as values):
            items = tuple(constant(value) for value in values)
        case Value(node=source, container=container) if (
            source.type.kind is Kind.LIST and container in (None, list)
        ):
            length, faults = invoke("len", [source], TypeError)
            comparison = "__lt__" if starred else "__ne__"
            faults.append(Fault(ValueError, apply(comparison, length, lit(before + after))))
            survivor, steps = context.guard(rest, faults)
            if survivor is not None:
                items = tuple(
                    Value(apply("__getitem__", source, lit(index))) for index in range(before)
                )
                if starred:
                    middle, _ = invoke(
                        "slice",
                        [source, lit(before), lit(-after or None, INT.or_none())],
                        TypeError,
                    )
                    items = (
                        *items,
                        Value(middle, list),
                        *(
                            Value(apply("__getitem__", source, lit(index)))
                            for index in range(-after, 0)
                        ),
                    )
                steps.append(survivor.push(*reversed(items)).advance())
            return steps
        case _:
            raise Unsupported("unpacking requires a list or tuple", context.where(state))
    if starred:
        raise Unsupported("starred unpacking requires a list", context.where(state))
    survivor, steps = context.guard(rest, [Fault(ValueError, lit(len(items) != before))])
    if survivor is not None:
        steps.append(survivor.push(*reversed(items)).advance())
    return steps


@handles("LIST_EXTEND", "SET_UPDATE")
def extend_container(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (item,), rest = state.pop()
    items, base = rest.pop(instruction.argval)
    try:
        members = constant_value(item)
        current = constant_value(items[0])
    except Unsupported:
        raise Unsupported(
            "list and set literal extension requires constants", context.where(state)
        ) from None
    if not isinstance(members, tuple | list | set | frozenset) or not isinstance(
        current, tuple | list | set | frozenset
    ):
        raise Unsupported("list and set literals", context.where(state))
    values = [*current, *members]
    value = set(values) if instruction.opname == "SET_UPDATE" else values
    return [base.push(constant(value), *items[1:]).advance()]
