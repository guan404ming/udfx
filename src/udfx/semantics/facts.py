from __future__ import annotations

from udfx.ir import Call, Col, Comprehension, If, ListValue, Lit, Node
from udfx.semantics.numbers import FLOAT_INT_MAX
from udfx.semantics.text import has_prefix_suffix_overlap
from udfx.types import STR, Kind, Type, fits

LENGTH_OPS = frozenset({"len", "set_size", "set_size_float"})


def bounded_divisor(node: Call) -> bool:
    divisor = node.arguments[1]
    return (
        isinstance(divisor, Lit)
        and isinstance(divisor.value, int)
        and divisor.value not in (0, -1)
        and fits(divisor.value, divisor.type)
    )


def float_exact_operands(node: Call) -> tuple[bool, ...]:
    return tuple(
        (
            isinstance(argument, Lit)
            and isinstance(argument.value, int)
            and -FLOAT_INT_MAX <= argument.value <= FLOAT_INT_MAX
        )
        or (isinstance(argument, Call) and argument.op.name in LENGTH_OPS)
        for argument in node.arguments
    )


def result_type(node: Call) -> Type:
    return node.type


def join_source_is_column(node: Call) -> bool:
    return isinstance(node.arguments[1], Col | Comprehension)


def argument_length(node: Call) -> int | None:
    return maximum_length(node.arguments[0])


def pattern_is_empty(node: Call) -> bool | None:
    pattern = node.arguments[1]
    return (
        pattern.value == "" if isinstance(pattern, Lit) and isinstance(pattern.value, str) else None
    )


def separator_is_none(node: Call) -> bool | None:
    separator = node.arguments[1]
    return separator.value is None if isinstance(separator, Lit) else None


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


def split_non_overlapping(node: Call) -> bool:
    separator = node.arguments[1]
    return (
        isinstance(separator, Lit)
        and isinstance(separator.value, str)
        and bool(separator.value)
        and not has_prefix_suffix_overlap(separator.value)
    )


def split_partitioned(node: Call) -> bool:
    return isinstance(node.arguments[0], Col | Lit) and split_non_overlapping(node)
