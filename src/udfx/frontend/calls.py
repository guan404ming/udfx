from __future__ import annotations

from collections.abc import Sequence

from udfx.errors import Unsupported
from udfx.frontend.values import Item, Method, Opaque, Value
from udfx.ir import Call, Node
from udfx.ops import Fault, apply, resolve


def node_of(item: Item) -> Node:
    if not isinstance(item, Value):
        raise Unsupported(f"expected a value, got {item}")
    return item.node


def qualified(obj: object) -> str:
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if name is None:
        raise Unsupported(f"cannot name callable {obj!r}")
    return name if module in (None, "builtins") else f"{module}.{name}"


def invoke(name: str, args: Sequence[Node], null_kind: type[Exception]) -> tuple[Call, list[Fault]]:
    kinds = tuple(arg.type.kind for arg in args)
    op = resolve(name, kinds)
    if op is None:
        raise Unsupported(f"no op {name}({', '.join(kinds)})")
    faults = [
        Fault(null_kind, apply("is_none", arg))
        for arg, param in zip(args, op.params, strict=True)
        if arg.type.nullable and not param.nullable
    ]
    faults.extend(op.faults(*args))
    return Call(op, tuple(args)), faults


def call(callee: Item, args: Sequence[Item]) -> tuple[Call, list[Fault]]:
    nodes = [node_of(arg) for arg in args]
    match callee:
        case Method(name=name):
            return invoke(f".{name}", nodes, AttributeError)
        case Opaque(obj=obj) if callable(obj):
            return invoke(qualified(obj), nodes, TypeError)
        case _:
            raise Unsupported(f"cannot call {callee}")
