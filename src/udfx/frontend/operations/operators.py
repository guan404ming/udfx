from __future__ import annotations

import operator
import re

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import Evaluation, always, single
from udfx.frontend.core.expressions import invoke, node_of, none_test, truth
from udfx.frontend.core.logic import conjunction, disjunction, fold, negate
from udfx.frontend.core.values import Item, Opaque, Row, Set, Slice, Tuple, Value
from udfx.frontend.operations.datetimes import datetime_binary
from udfx.frontend.operations.formatting import format_percent
from udfx.frontend.operations.sets import set_operator
from udfx.frontend.resolve.constants import (
    GENERICS,
    annotation,
    constant,
    constant_value,
    is_scalar,
    lookup,
    type_comparison,
)
from udfx.ir import FALSE, TRUE, Lit, Node, lit
from udfx.ops import Fault
from udfx.semantics.lists import matches_itself
from udfx.types import Kind

BINARY_OPS: dict[str, str] = {
    "+": "__add__",
    "-": "__sub__",
    "*": "__mul__",
    "/": "__truediv__",
    "//": "__floordiv__",
    "%": "__mod__",
    "**": "__pow__",
    "&": "and",
    "|": "or",
}

COMPARE_OPS: dict[str, str] = {
    "<": "__lt__",
    "<=": "__le__",
    "==": "__eq__",
    "!=": "__ne__",
    ">": "__gt__",
    ">=": "__ge__",
}


def binary_operation(symbol: str, left: Item, right: Item, where: str) -> Evaluation:
    base = symbol.removesuffix("=")
    if symbol.endswith("=") and (
        isinstance(left, Set) or (isinstance(left, Value) and left.node.type.kind is Kind.LIST)
    ):
        container = "set" if isinstance(left, Set) else "list"
        raise Unsupported(f"in-place {container} mutation", where)
    special = set_operator(base, left, right) or datetime_binary(base, left, right)
    if special is not None:
        item, faults = special
        return single(item, faults)
    if base == "|" and any(
        isinstance(item, Opaque) and isinstance(item.value, re.RegexFlag) for item in (left, right)
    ):
        first, second = constant_value(left), constant_value(right)
        if isinstance(first, int) and isinstance(second, int):
            return single(constant(first | second))
    if base == "|" and not isinstance(left, Value):
        return single(annotation(operator.or_, left, right))
    if base == "%" and isinstance(left, Value) and left.node.type.kind is Kind.STR:
        return string_format(left, right, where)
    name = BINARY_OPS.get(base)
    if name is None:
        raise Unsupported(f"binary operator {symbol}", where)
    node, faults = invoke(name, [node_of(left), node_of(right)], TypeError)
    return single(Value(node), faults)


def string_format(left: Value, right: Item, where: str) -> Evaluation:
    if isinstance(right, Value) and right.container in (list, set, frozenset):
        raise Unsupported("constant container outside an in test", where)
    if isinstance(right, Value) and right.container is not tuple:
        node_of(right)
    match right:
        case Tuple(items=items):
            arguments = items
        case Value(node=Lit(value=tuple() as values)) | Opaque(value=tuple() as values):
            arguments = tuple(constant(value) for value in values)
        case _:
            arguments = (right,)
    node, faults = format_percent(left.node, [node_of(item) for item in arguments])
    return single(Value(fold(node)), faults)


def comparison(symbol: str, left: Item, right: Item, where: str) -> Evaluation:
    name = COMPARE_OPS.get(symbol)
    if name is None:
        raise Unsupported(f"comparison {symbol}", where)
    if name in ("__eq__", "__ne__"):
        if all(isinstance(item, Value) and isinstance(item.node, Lit) for item in (left, right)):
            return single(
                constant(getattr(operator, name)(constant_value(left), constant_value(right)))
            )
        test = type_comparison(left, right)
        if test is not None:
            return single(Value(negate(test) if name == "__ne__" else test))
    node, faults = invoke(name, [node_of(left), node_of(right)], TypeError)
    return single(Value(node), faults)


def logical_not(item: Item) -> Evaluation:
    test, faults = truth(item)
    return single(Value(negate(test)), faults)


def negation(item: Item) -> Evaluation:
    node, faults = invoke("__neg__", [node_of(item)], TypeError)
    return single(Value(node), faults)


def identity(left: Item, right: Item, negated: bool, where: str) -> Evaluation:
    if is_none_constant(right):
        test = none_test(left)
    elif is_none_constant(left):
        test = none_test(right)
    else:
        test = type_comparison(left, right)
        if test is None:
            raise Unsupported("identity check against a value other than None", where)
    return single(Value(negate(test) if negated else test))


def membership(item: Item, container: Item, negated: bool) -> Evaluation:
    if isinstance(container, Opaque) and isinstance(container.value, dict):
        node, faults = lookup(dict.fromkeys(container.value, True), node_of(item), lit(False))
    elif isinstance(container, Tuple):
        node, faults = tuple_membership(item, container)
    else:
        members = container.node if isinstance(container, Value | Set) else node_of(container)
        node, faults = invoke("__contains__", [members, node_of(item)], TypeError)
    return single(Value(negate(node) if negated else node), faults)


def subscript(container: Item, index: Item, where: str) -> Evaluation:
    if isinstance(container, Opaque) and isinstance(container.value, tuple):
        container = Tuple(tuple(constant(value) for value in container.value))
    if isinstance(container, Tuple):
        position = constant_value(index)
        if not isinstance(position, int):
            raise Unsupported("tuple index must be an integer literal", where)
        try:
            return single(container.items[position])
        except IndexError:
            return always(IndexError)
    if isinstance(container, Row):
        if not (
            isinstance(index, Value)
            and isinstance(index.node, Lit)
            and isinstance(index.node.value, str | int)
        ):
            raise Unsupported("row access requires a constant string or integer key", where)
        try:
            return single(container.column(index.node.value))
        except IndexError:
            return always(IndexError)
    if isinstance(container, Opaque) and any(container.value is generic for generic in GENERICS):
        return single(annotation(operator.getitem, container, index))
    if isinstance(index, Opaque) and isinstance(index.value, slice):
        bounds = (index.value.start, index.value.stop, index.value.step)
        index = Slice(tuple(node_of(constant(bound)) for bound in bounds))
    if isinstance(index, Slice):
        node, faults = invoke("slice", [node_of(container), *index.bounds], TypeError)
    elif isinstance(container, Opaque):
        node, faults = lookup(container.value, node_of(index))
    else:
        node, faults = invoke("__getitem__", [node_of(container), node_of(index)], TypeError)
    return single(Value(node), faults)


def build_slice(items: tuple[Item, ...], where: str) -> Evaluation:
    bounds = tuple(node_of(item) for item in items)
    if len(bounds) == 3 and not isinstance(bounds[2], Lit):
        raise Unsupported("slice step must be a literal", where)
    return single(Slice(bounds))


def cut(container: Item, start: Item, stop: Item) -> Evaluation:
    node, faults = invoke("slice", [node_of(container), node_of(start), node_of(stop)], TypeError)
    return single(Value(fold(node)), faults)


def is_none_constant(item: Item) -> bool:
    return isinstance(item, Value) and isinstance(item.node, Lit) and item.node.value is None


def tuple_membership(item: Item, container: Tuple) -> tuple[Node, list[Fault]]:
    if any(not isinstance(candidate, Tuple) for candidate in container.items):
        raise Unsupported("membership in a dynamic tuple")
    if isinstance(item, Value) and item.container is not tuple:
        return FALSE, []
    if any(
        not is_scalar(constant_value(part))
        for candidate in container.items
        if isinstance(candidate, Tuple)
        for part in candidate.items
    ):
        raise Unsupported("tuple membership requires scalar constants")
    candidates = tuple(constant_value(candidate) for candidate in container.items)
    try:
        value = constant_value(item)
    except Unsupported:
        if not isinstance(item, Tuple):
            raise
    else:
        if not isinstance(value, tuple) or not all(is_scalar(member) for member in value):
            raise Unsupported("tuple membership requires scalar values")
        return lit(value in candidates), []
    result: Node = FALSE
    faults: list[Fault] = []
    for candidate in container.items:
        if not isinstance(candidate, Tuple):
            continue
        matched, pending = tuple_match(item, candidate)
        faults.extend(pending)
        result = disjunction(result, matched)
    return result, faults


def tuple_match(item: Tuple, candidate: Tuple) -> tuple[Node, list[Fault]]:
    if len(candidate.items) != len(item.items):
        return FALSE, []
    matched: Node = TRUE
    faults: list[Fault] = []
    for left, right in zip(item.items, candidate.items, strict=True):
        if not matches_itself(constant_value(right)):
            raise Unsupported("non-reflexive values in dynamic tuple membership")
        test, pending = invoke("__eq__", [node_of(left), node_of(right)], TypeError)
        matched = conjunction(matched, test)
        faults.extend(pending)
    return matched, faults
