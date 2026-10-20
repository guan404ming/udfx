from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import cast

from udfx.errors import Unsupported
from udfx.frontend.logic import fold
from udfx.frontend.values import Item, Method, Opaque, Value, describe, named
from udfx.ir import FALSE, Call, Lit, Node, lit
from udfx.ops import Fault, Op, apply, matching
from udfx.types import BOOL, Kind, type_of

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


def overload(name: str, arguments: Sequence[Node]) -> tuple[Op, list[Node]]:
    types = [argument.type for argument in arguments]
    candidates = matching(name, types)
    others = {type.kind for type in types if type.kind is not Kind.NONE}
    preferred = [
        op
        for op in candidates
        if all(
            parameter.kind in others
            for type, parameter in zip(types, op.parameters, strict=True)
            if type.kind is Kind.NONE
        )
    ]
    chosen = preferred or candidates
    if not chosen:
        raise Unsupported(f"no op {name}({', '.join(str(type.strict()) for type in types)})")
    op = chosen[0]
    typed = [
        Lit(None, parameter) if argument.type.kind is Kind.NONE else argument
        for argument, parameter in zip(arguments, op.parameters, strict=True)
    ]
    return op, typed


def invoke(
    name: str, arguments: Sequence[Node], first_null_kind: type[Exception]
) -> tuple[Node, list[Fault]]:
    op, arguments = overload(name, arguments)
    call = Call(op, tuple(arguments))
    folded = fold(call)
    if isinstance(folded, Lit) and folded.type.kind is not Kind.LIST:
        return folded, []
    faults = [
        Fault(first_null_kind if index == 0 else TypeError, apply("is_none", argument))
        for index, (argument, parameter) in enumerate(zip(arguments, op.parameters, strict=True))
        if argument.type.nullable and not parameter.nullable
    ]
    faults.extend(op.faults(*arguments))
    return (call if isinstance(folded, Lit) else folded), faults


def evaluated(nodes: Sequence[Node], run: Callable[[list[object]], object]) -> Lit | None:
    values = [node.value for node in nodes if isinstance(node, Lit)]
    if len(values) < len(nodes):
        return None
    try:
        value = run(values)
    except Exception:
        return None
    type = type_of(value)
    return None if type is None else Lit(value, type)


def call(callee: Item, arguments: Sequence[Item]) -> tuple[Node, list[Fault]]:
    nodes = [node_of(argument) for argument in arguments]
    match callee:
        case Method(name=name):
            folded = evaluated(nodes, lambda values: getattr(values[0], name)(*values[1:]))
            return (folded, []) if folded is not None else invoke(f".{name}", nodes, AttributeError)
        case Opaque(value=value):
            name = op_name(value)
            function = cast("Callable[..., object]", value)
            folded = evaluated(nodes, lambda values: function(*values))
            return (folded, []) if folded is not None else invoke(name, nodes, TypeError)
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
        case Opaque(value=value):
            return lit(value is None)
        case _:
            raise Unsupported(f"None test of {describe(item)}")
