from __future__ import annotations

import math
from collections.abc import Sequence

from udfx.errors import Unsupported
from udfx.frontend.values import Item, Method, Opaque, Value, describe, named
from udfx.ir import FALSE, Call, Lit, Node, lit
from udfx.ops import Fault, Op, all_ops, apply, resolve
from udfx.types import BOOL, Kind

CALLABLES: tuple[tuple[object, str], ...] = (
    (abs, "abs"),
    (bool, "bool"),
    (float, "float"),
    (int, "int"),
    (len, "len"),
    (max, "max"),
    (min, "min"),
    (round, "round"),
    (str, "str"),
    (math.ceil, "math.ceil"),
    (math.floor, "math.floor"),
    (math.isinf, "math.isinf"),
    (math.isnan, "math.isnan"),
    (math.log10, "math.log10"),
    (math.sqrt, "math.sqrt"),
)


def node_of(item: Item) -> Node:
    if not isinstance(item, Value):
        raise Unsupported(f"expected a value, got {describe(item)}")
    if isinstance(item.node, Lit) and item.node.type.kind is Kind.LIST:
        raise Unsupported(f"constant container {item.node.value!r} outside an in test")
    return item.node


def op_name(callee: object) -> str:
    for known, name in CALLABLES:
        if known is callee:
            return name
    raise Unsupported(f"unknown callable {named(callee)}")


def accepts(op: Op, kinds: tuple[Kind, ...]) -> bool:
    return len(op.params) == len(kinds) and all(
        kind is parameter.kind or (kind is Kind.NONE and parameter.nullable)
        for kind, parameter in zip(kinds, op.params, strict=True)
    )


def overload(name: str, arguments: Sequence[Node]) -> tuple[Op, list[Node]]:
    kinds = tuple(argument.type.kind for argument in arguments)
    op = resolve(name, kinds)
    if op is None:
        candidates = [op for op in all_ops() if op.name == name and accepts(op, kinds)]
        if len(candidates) != 1:
            raise Unsupported(f"no op {name}({', '.join(kinds)})")
        (op,) = candidates
    typed = [
        Lit(None, parameter) if argument.type.kind is Kind.NONE else argument
        for argument, parameter in zip(arguments, op.params, strict=True)
    ]
    return op, typed


def invoke(
    name: str, arguments: Sequence[Node], first_null_kind: type[Exception]
) -> tuple[Call, list[Fault]]:
    op, arguments = overload(name, arguments)
    faults = [
        Fault(first_null_kind if index == 0 else TypeError, apply("is_none", argument))
        for index, (argument, parameter) in enumerate(zip(arguments, op.params, strict=True))
        if argument.type.nullable and not parameter.nullable
    ]
    faults.extend(op.faults(*arguments))
    return Call(op, tuple(arguments)), faults


def call(callee: Item, arguments: Sequence[Item]) -> tuple[Call, list[Fault]]:
    nodes = [node_of(argument) for argument in arguments]
    match callee:
        case Method(name=name):
            return invoke(f".{name}", nodes, AttributeError)
        case Opaque(obj=obj):
            return invoke(op_name(obj), nodes, TypeError)
        case _:
            raise Unsupported(f"cannot call {describe(callee)}")


def truth(item: Item) -> tuple[Node, list[Fault]]:
    match item:
        case Value(node=Lit(value=value)):
            return lit(bool(value)), []
        case Value(node=node) if node.type == BOOL:
            return node, []
        case Value(node=node):
            return invoke("bool", [node], TypeError)
        case _:
            raise Unsupported(f"truth of {describe(item)}")


def none_test(item: Item) -> Node:
    match item:
        case Value(node=Lit(value=value)):
            return lit(value is None)
        case Value(node=node) if node.type.nullable:
            return apply("is_none", node)
        case Value():
            return FALSE
        case Opaque(obj=obj):
            return lit(obj is None)
        case _:
            raise Unsupported(f"None test of {describe(item)}")
