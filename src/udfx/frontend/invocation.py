from __future__ import annotations

from collections.abc import Callable, Sequence

from udfx.errors import Unsupported
from udfx.frontend.logic import fold
from udfx.frontend.values import Item, Value, describe
from udfx.ir import TRUE, Call, Lit, Node, lit
from udfx.ops import Fault, Op, apply, matching
from udfx.semantics.datetimes import format_parts, parse_parts
from udfx.types import Kind, type_of


def node_of(item: Item) -> Node:
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
        return node, [Fault(TypeError, apply("is_none", node))] if node.type.nullable else []
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
