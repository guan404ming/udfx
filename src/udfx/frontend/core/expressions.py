from __future__ import annotations

from collections.abc import Callable, Sequence

from udfx.errors import Unsupported
from udfx.frontend.core.json import dynamic
from udfx.frontend.core.logic import fold, negate
from udfx.frontend.core.tracing import EFFECTS
from udfx.frontend.core.values import (
    Item,
    Json,
    Match,
    Opaque,
    Set,
    Tuple,
    Value,
    describe,
)
from udfx.ir import FALSE, TRUE, Call, Lit, Node, lit
from udfx.ops import Fault, Op, apply, matching
from udfx.semantics.datetimes import format_parts, parse_parts
from udfx.types import BOOL, Kind, type_of


def node_of(item: Item) -> Node:
    if isinstance(item, Json):
        return dynamic(item.node)
    if not isinstance(item, Value):
        raise Unsupported(f"expected a value, got {describe(item)}")
    if item.container in (tuple, set, frozenset):
        raise Unsupported(f"constant container {describe(item)} outside an in test")
    return item.node


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
    if name == "os.fspath" and len(arguments) == 1:
        node = arguments[0]
        if node.type.kind is not Kind.STR:
            return lit(""), [Fault(TypeError, TRUE)]
        return node, nullable_faults(node)
    if (
        name in {".center", ".ljust", ".rjust"}
        and len(arguments) == 3
        and not isinstance(arguments[2], Lit)
    ):
        raise Unsupported("fill character must be a literal")
    if name in {".strftime", "datetime.strptime"}:
        if (
            len(arguments) != 2
            or not isinstance(arguments[1], Lit)
            or not isinstance(arguments[1].value, str)
        ):
            raise Unsupported("datetime format must be a string literal")
        if name == "datetime.strptime":
            parse_parts(arguments[1].value)
        else:
            format_parts(arguments[1].value)
    op, arguments = overload(name, arguments)
    effects = EFFECTS.get()
    if not op.deterministic and effects is not None:
        effects.add(name)
    call = Call(op, tuple(arguments))
    folded = fold(call)
    if (
        isinstance(folded, Lit)
        and folded.type.kind is not Kind.LIST
        and all(isinstance(argument, Lit) for argument in arguments)
    ):
        return folded, []
    faults = [
        fault
        for index, (argument, parameter) in enumerate(zip(arguments, op.parameters, strict=True))
        if not parameter.nullable
        for fault in nullable_faults(argument, first_null_kind if index == 0 else TypeError)
    ]
    faults.extend(op.faults(*arguments))
    return (call if isinstance(folded, Lit) and folded.type.kind is Kind.LIST else folded), faults


def nullable_faults(node: Node, kind: type[Exception] = TypeError) -> list[Fault]:
    return [Fault(kind, apply("is_none", node))] if node.type.nullable else []


def evaluated(nodes: Sequence[Node], run: Callable[[list[object]], object]) -> Lit | None:
    values = [
        list(node.value)
        if node.type.kind is Kind.LIST and isinstance(node.value, tuple)
        else node.value
        for node in nodes
        if isinstance(node, Lit)
    ]
    if len(values) < len(nodes):
        return None
    try:
        value = run(values)
    except Exception:
        return None
    type = type_of(value)
    return None if type is None else Lit(value, type)


def truth(item: Item) -> tuple[Node, list[Fault]]:
    match item:
        case Tuple(items=items):
            return lit(bool(items)), []
        case Match():
            return negate(none_test(item)), []
        case Set(node=node):
            return invoke("bool", [node], TypeError)
        case Json(node=node):
            return invoke("json.truth", [dynamic(node)], TypeError)
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
        case Json(node=node):
            return apply("json.none", dynamic(node))
        case Match(node=node):
            return none_test(Value(node))
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


def returned(item: Item, where: str) -> Node:
    if (
        isinstance(item, Tuple)
        or (isinstance(item, Value) and item.container is tuple)
        or (isinstance(item, Opaque) and isinstance(item.value, tuple))
    ):
        raise Unsupported("tuple returns are unsupported; return a scalar or list", where)
    return node_of(item)
