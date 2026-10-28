from __future__ import annotations

from udfx.ir import TRUE, Call, Col, Comprehension, If, ListValue, Lit, Node, lit
from udfx.ops.registry import apply
from udfx.semantics import json_paths
from udfx.semantics.assumptions import CONDITIONS
from udfx.semantics.bindings import source as bound_source
from udfx.semantics.numbers import FLOAT_INT_MAX
from udfx.semantics.ranges import LENGTH_OPS, integer_bounds, maximum_length, proven
from udfx.semantics.text import has_prefix_suffix_overlap
from udfx.semantics.text_predicates import ASCII_PREDICATES
from udfx.types import INT64_MAX, INT64_MIN, STR, Kind, Type, fits


def element_predicate(source: Node, element: Node) -> Node:
    source = bound_source(source)
    if isinstance(source, Call) and source.op.name in {".split", ".rsplit"}:
        separator = source.arguments[1] if len(source.arguments) > 1 else lit(None)
        limit = source.arguments[2] if len(source.arguments) > 2 else lit(-1)
        if (
            isinstance(separator, Lit)
            and separator.value is None
            and isinstance(limit, Lit)
            and isinstance(limit.value, int)
            and limit.value < 0
        ):
            return apply("not", apply("__contains__", element, lit("\n")))
    if not isinstance(source, Call) or source.op.name != "json.tokens":
        return TRUE
    predicate = apply("and", apply("json.valid", element), apply("json.string_native", element))
    for kind in (str, list, dict, int, float, bool, type(None)):
        check = apply("json.is_type", element, lit(kind.__name__))
        predicate = apply("and", predicate, check if kind is str else apply("not", check))
    return predicate


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


def nullable_arguments(node: Call) -> bool:
    return any(
        argument.type.nullable and proven(apply("is_none", argument)) is not False
        for argument in node.arguments
    )


def join_source_is_column(node: Call) -> bool:
    return isinstance(node.arguments[1], Col | Comprehension)


def argument_length(node: Call) -> int | None:
    return maximum_length(node.arguments[0])


def integer_is_valid(node: Call) -> bool:
    return any(
        truth
        and isinstance(predicate, Call)
        and predicate.op.name == "int_parses"
        and predicate.arguments == node.arguments
        for predicate, truth in CONDITIONS.get()
    )


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


def json_default_is_none(node: Call) -> bool:
    value = node.arguments[2]
    return isinstance(value, Lit) and value.value == "null"


def json_argument_validated(node: Call, *, index: int = 0) -> bool:
    seen: dict[int, bool] = {}

    def validated(value: Node) -> bool:
        if id(value) not in seen:
            seen[id(value)] = check(value)
        return seen[id(value)]

    def check(value: Node) -> bool:
        value = bound_source(value)
        if isinstance(value, Lit) and value.value == "null":
            return True
        if isinstance(value, If):
            return validated(value.then) and validated(value.orelse)
        if isinstance(value, Call):
            if value.op.name in {"json.value", "json.get", "json.item", "json.encode"}:
                return True
            if value.op.name == "json.text":
                return validated(value.arguments[0])
            if value.op.name == "__getitem__":
                source = bound_source(value.arguments[0])
                return isinstance(source, Call) and source.op.name == "json.items"
        return any(
            truth
            and isinstance(predicate, Call)
            and predicate.op.name == "json.valid"
            and predicate.arguments == (value,)
            for predicate, truth in CONDITIONS.get()
        )

    return validated(node.arguments[index])


def json_string_is_native(node: Call) -> bool:
    value = json_paths.source(node.arguments[0])
    return value.type.kind is Kind.STR and proven(apply("json.string_native", value)) is True


def json_argument_type(node: Call) -> str | None:
    return json_paths.container(node)


def json_path_source(node: Call) -> Node | None:
    return json_paths.indexed_path(node)[0]


def json_path_indexes(node: Call) -> tuple[int, ...]:
    return json_paths.indexed_path(node)[1]
