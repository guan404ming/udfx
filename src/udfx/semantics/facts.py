from __future__ import annotations

from udfx.ir import Call, If, ListValue, Lit, Node
from udfx.types import STR, Kind, Type, fits


def bounded_divisor(node: Call) -> bool:
    divisor = node.arguments[1]
    return (
        isinstance(divisor, Lit)
        and isinstance(divisor.value, int)
        and divisor.value not in (0, -1)
        and fits(divisor.value, divisor.type)
    )


def result_type(node: Call) -> Type:
    return node.type


def argument_length(node: Call) -> int | None:
    return maximum_length(node.arguments[0])


def maximum_length(node: Node) -> int | None:
    if isinstance(node, Lit) and isinstance(node.value, str):
        return len(node.value)
    if not isinstance(node, Call):
        return None
    if node.op.name in {".strip", ".lstrip", ".rstrip"}:
        return maximum_length(node.arguments[0])
    if node.op.name != "slice":
        return None
    text, start, stop, *steps = node.arguments
    if steps and (not isinstance(steps[0], Lit) or steps[0].value not in (None, 1)):
        return maximum_length(text)
    bounds = [maximum_length(text)]
    if isinstance(start, Lit) and isinstance(start.value, int) and start.value < 0:
        bounds.append(-start.value)
    if isinstance(stop, Lit) and isinstance(stop.value, int) and stop.value >= 0:
        begin = start.value if isinstance(start, Lit) and isinstance(start.value, int) else 0
        bounds.append(max(stop.value - max(begin, 0), 0))
    known = [bound for bound in bounds if bound is not None]
    return min(known) if known else None


def minimum_length(node: Node) -> int:
    if isinstance(node, Lit) and isinstance(node.value, str | tuple | list):
        return len(node.value)
    if isinstance(node, ListValue):
        return len(node.items)
    if isinstance(node, If):
        return min(minimum_length(node.then), minimum_length(node.orelse))
    if isinstance(node, Call):
        if node.op.name in {".split", ".rsplit"} and len(node.arguments) > 1:
            return int(node.arguments[1].type == STR)
        if node.op.name == "__add__" and node.type.kind in {Kind.STR, Kind.LIST}:
            return sum(map(minimum_length, node.arguments))
    return 0
