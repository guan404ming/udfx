from __future__ import annotations

from udfx.ir import Call, Col, Comprehension, If, ListValue, Lit, Node
from udfx.semantics.assumptions import CONDITIONS
from udfx.semantics.numbers import FLOAT_INT_MAX
from udfx.semantics.ranges import LENGTH_OPS, integer_bounds, maximum_length
from udfx.semantics.text import has_prefix_suffix_overlap
from udfx.semantics.text_predicates import ASCII_PREDICATES
from udfx.types import INT64_MAX, INT64_MIN, STR, Kind, Type, fits


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


def ascii_predicate_result(node: Call) -> bool | None:
    pattern = ASCII_PREDICATES.get(node.op.name.removeprefix("."))
    if pattern is None:
        return None
    for predicate, truth in CONDITIONS.get():
        if (
            isinstance(predicate, Call)
            and predicate.op.name.startswith(".")
            and predicate.arguments == node.arguments
            and ASCII_PREDICATES.get(predicate.op.name[1:]) == pattern
        ):
            return truth
    return None


def simple_arguments(node: Call) -> bool:
    return all(isinstance(argument, Col | Lit) for argument in node.arguments)


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


def split_separator(node: Call) -> str | None:
    source = node.arguments[0]
    while isinstance(source, Call) and source.op.name in {"slice", "list"}:
        source = source.arguments[0]
    if not isinstance(source, Call) or source.op.name not in {".split", ".rsplit"}:
        return None
    if len(source.arguments) != 2:
        return None
    separator = source.arguments[1]
    if (
        isinstance(separator, Lit)
        and isinstance(separator.value, str)
        and len(separator.value) == 1
    ):
        return separator.value
    return None


def integer_fits(node: Call, *, size_limit: int) -> bool:
    bounds = integer_bounds(node, size_limit)
    return bounds is not None and INT64_MIN <= bounds[0] <= bounds[1] <= INT64_MAX


def argument_maximum(node: Call) -> int | None:
    bounds = integer_bounds(node.arguments[1], INT64_MAX)
    return bounds[1] if bounds is not None else None
