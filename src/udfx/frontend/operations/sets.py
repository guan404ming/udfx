from __future__ import annotations

from collections.abc import Mapping, Sequence

from udfx.errors import Unsupported
from udfx.frontend.core.expressions import invoke
from udfx.frontend.core.values import Item, Set, Value
from udfx.frontend.operations.builtins import iterable_node
from udfx.frontend.resolve.constants import constant, constant_value, is_scalar
from udfx.ops import Fault
from udfx.semantics.lists import SET_OPERATIONS, SET_PREDICATES

SET_OPERATORS = {"&": "intersection", "|": "union", "-": "difference", "^": "symmetric_difference"}


def frozen_set(arguments: Sequence[Item]) -> tuple[Item, list[Fault]]:
    if len(arguments) > 1:
        raise TypeError("frozenset accepts at most one argument")
    source = constant_value(arguments[0]) if arguments else ()
    if not isinstance(source, tuple | list | set | frozenset | str):
        raise Unsupported("frozenset requires a constant sequence")
    if type(source) not in (tuple, list, set, frozenset, str) or not all(map(is_scalar, source)):
        raise Unsupported("frozenset requires scalar constants")
    return constant(frozenset(source)), []


def set_binary(operation: str, left: Set, right: Set) -> tuple[Item, list[Fault]]:
    node, faults = invoke(f"set_{operation}", [left.node, right.node], TypeError)
    return (
        Value(node) if operation in SET_PREDICATES else Set(node, left.dynamic or right.dynamic)
    ), faults


def set_operator(symbol: str, left: Item, right: Item) -> tuple[Item, list[Fault]] | None:
    if isinstance(left, Set) and isinstance(right, Set) and symbol in SET_OPERATORS:
        return set_binary(SET_OPERATORS[symbol], left, right)
    return None


def set_method(
    name: str, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> tuple[Item, list[Fault]]:
    if keywords or name not in (*SET_OPERATIONS, *SET_PREDICATES):
        raise Unsupported(f"set method {name}")
    if name in (*SET_PREDICATES, "symmetric_difference") and len(arguments) != 2:
        raise TypeError(f"{name} requires one argument")
    result = arguments[0]
    faults: list[Fault] = []
    for argument in arguments[1:]:
        if not isinstance(result, Set):
            raise Unsupported(f"set method {name}")
        if isinstance(argument, Set):
            source = argument.node
        else:
            source = iterable_node(argument)
            source, pending = invoke("list", [source], TypeError)
            faults.extend(pending)
        result, pending = set_binary(
            name, result, Set(source, isinstance(argument, Set) and argument.dynamic)
        )
        faults.extend(pending)
    return result, faults
