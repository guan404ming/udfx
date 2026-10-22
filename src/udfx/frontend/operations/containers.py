from __future__ import annotations

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import Evaluation, single
from udfx.frontend.core.expressions import invoke, node_of
from udfx.frontend.core.values import Item, Opaque, Set, Tuple, Value
from udfx.frontend.resolve.constants import constant, constant_value, scalar_map
from udfx.ir import ListValue, Lit, Node, lit
from udfx.ops import Fault, apply
from udfx.types import INT, STR, Kind, list_of


def build_map(items: tuple[Item, ...]) -> Evaluation:
    constants = [constant_value(item) for item in items]
    scalar_map((key, None) for key in constants[::2])
    return single(Opaque(dict(zip(constants[::2], constants[1::2], strict=True))))


def build_keys_map(items: tuple[Item, ...], where: str) -> Evaluation:
    *values, keys = items
    names = constant_value(keys)
    if not isinstance(names, tuple):
        raise Unsupported(f"dict keys {names!r}", where)
    return single(Opaque(dict(zip(names, map(constant_value, values), strict=True))))


def build_tuple(items: tuple[Item, ...]) -> Evaluation:
    try:
        values = tuple(constant_value(item) for item in items)
    except Unsupported:
        return single(Tuple(items))
    return single(constant(values))


def build_set(items: tuple[Item, ...], where: str) -> Evaluation:
    try:
        return single(constant({constant_value(item) for item in items}))
    except Unsupported:
        nodes = tuple(node_of(item) for item in items)
        if any(
            node.type != nodes[0].type
            or node.type.kind not in {Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL}
            for node in nodes
        ):
            raise Unsupported("set requires matching scalar items", where) from None
        return single(Set(ListValue(nodes, list_of(nodes[0].type))))


def build_list(items: tuple[Item, ...], where: str) -> Evaluation:
    nodes = tuple(node_of(item) for item in items)
    if nodes and any(node.type != nodes[0].type for node in nodes):
        raise Unsupported("list items must share one type", where)
    result = list_of(nodes[0].type if nodes else STR)
    value = (
        Lit(tuple(node.value for node in nodes if isinstance(node, Lit)), result)
        if all(isinstance(node, Lit) for node in nodes)
        else ListValue(nodes, result)
    )
    return single(Value(value, list))


def unpack(item: Item, before: int, after: int, starred: bool, where: str) -> Evaluation:
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
            return unpack_list(source, before, after, starred)
        case _:
            raise Unsupported("unpacking requires a list or tuple", where)
    if starred:
        raise Unsupported("starred unpacking requires a list", where)
    return Evaluation(items, [Fault(ValueError, lit(len(items) != before))])


def unpack_list(source: Node, before: int, after: int, starred: bool) -> Evaluation:
    length, faults = invoke("len", [source], TypeError)
    comparison = "__lt__" if starred else "__ne__"
    faults.append(Fault(ValueError, apply(comparison, length, lit(before + after))))
    items: tuple[Item, ...] = tuple(
        Value(apply("__getitem__", source, lit(index))) for index in range(before)
    )
    if starred:
        middle, _ = invoke(
            "slice", [source, lit(before), lit(-after or None, INT.or_none())], TypeError
        )
        items = (
            *items,
            Value(middle, list),
            *(Value(apply("__getitem__", source, lit(index))) for index in range(-after, 0)),
        )
    return Evaluation(items, faults)


def extend(current: Item, source: Item, kind: str, where: str) -> Evaluation:
    try:
        members = constant_value(source)
        existing = constant_value(current)
    except Unsupported:
        raise Unsupported("list and set literal extension requires constants", where) from None
    sequences = (tuple, list, set, frozenset)
    if not isinstance(members, sequences) or not isinstance(existing, sequences):
        raise Unsupported("list and set literals", where)
    values = [*existing, *members]
    return single(constant(set(values) if kind == "set" else values))
