from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from udfx.errors import Unsupported
from udfx.frontend.core.expressions import evaluated, invoke, node_of, nullable_faults, truth
from udfx.frontend.core.logic import disjunction, fold
from udfx.frontend.core.reductions import reduce_generator
from udfx.frontend.core.values import (
    Duration,
    Generator,
    Item,
    Method,
    Opaque,
    Set,
    Tuple,
    Value,
    describe,
    named,
    require_bounded,
)
from udfx.frontend.operations.formatting import format_template
from udfx.frontend.resolve.callables import op_name
from udfx.frontend.resolve.functions import bind
from udfx.ir import FALSE, TRUE, Comprehension, Element, Lit, Node, lit
from udfx.ops import Fault
from udfx.semantics.datetimes import MICROSECONDS_PER_SECOND
from udfx.semantics.translation import translation_table
from udfx.types import BOOL, INT, STR, Kind, list_of


def iterable_node(item: Item) -> Node:
    if isinstance(item, Generator):
        return item.node
    if isinstance(item, Set):
        return sequence_call("list", item)[0]
    if isinstance(item, Value) and item.container in (tuple, list):
        return item.node
    return node_of(item)


def keyword_arguments(
    function: Callable[..., object], arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> list[Item]:
    try:
        signature = inspect.signature(function, follow_wrapped=False)
    except (TypeError, ValueError) as error:
        raise Unsupported(f"keyword arguments for {named(function)}: no signature") from error
    bindings = bind(signature, arguments, keywords)
    positional: list[Item] = []
    remaining = set(bindings)
    for name, parameter in signature.parameters.items():
        if not remaining:
            break
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise Unsupported(f"keyword arguments for {named(function)}")
        positional.append(bindings[name] if name in bindings else Value(lit(parameter.default)))
        remaining.discard(name)
    return positional


def tuple_affixes(name: str, arguments: Sequence[Item]) -> tuple[Node, list[Fault]] | None:
    if len(arguments) != 2:
        return None
    match arguments[1]:
        case Value(node=Lit(value=prefixes), container=container) if container is tuple:
            pass
        case Opaque(value=tuple() as prefixes):
            pass
        case _:
            return None
    if not isinstance(prefixes, tuple) or not all(isinstance(prefix, str) for prefix in prefixes):
        raise Unsupported("affixes must be a tuple of string literals")
    owner = node_of(arguments[0])
    result: Node = FALSE
    faults: list[Fault] = []
    for prefix in prefixes:
        test, pending = invoke(f".{name}", [owner, lit(prefix)], AttributeError)
        result = disjunction(result, test)
        faults.extend(pending)
    if not prefixes:
        _, faults = invoke(f".{name}", [owner, lit("")], AttributeError)
    return result, faults


def extremum(name: str, nodes: Sequence[Node]) -> tuple[Node, list[Fault]]:
    result = nodes[0]
    faults: list[Fault] = []
    for node in nodes[1:]:
        result, pending = invoke(name, [result, node], TypeError)
        faults.extend(pending)
    return result, faults


def call(
    callee: Item, arguments: Sequence[Item], keywords: Mapping[str, Item] | None = None
) -> tuple[Node, list[Fault]]:
    if keywords:
        match callee:
            case Method(name=name):
                owner = node_of(arguments[0])
                owner_type = {Kind.STR: str, Kind.LIST: list}.get(owner.type.kind)
                function = getattr(owner_type, name, None)
            case Opaque(value=value):
                op_name(value)
                function = value
            case _:
                raise Unsupported(f"cannot call {describe(callee)}")
        arguments = keyword_arguments(cast("Callable[..., object]", function), arguments, keywords)
    if isinstance(callee, Opaque) and len(arguments) == 1:
        name = op_name(callee.value)
        if isinstance(arguments[0], Set | Tuple) or name in {"list", "sorted", "any", "all", "sum"}:
            return sequence_call(name, arguments[0])
    match callee:
        case Method(name=name):
            return method_call(name, arguments)
        case Opaque(value=value):
            name = op_name(value)
            nodes = [node_of(argument) for argument in arguments]
            if name == "int" and len(nodes) == 1 and nodes[0].type.kind is Kind.INT:
                return nodes[0], nullable_faults(nodes[0])
            if name == "datetime.strptime":
                return invoke(name, nodes, TypeError)
            if name in {"min", "max"} and len(nodes) > 2:
                return extremum(name, nodes)
            function = cast("Callable[..., object]", value)
            folded = evaluated(nodes, lambda values: function(*values))
            return (folded, []) if folded is not None else invoke(name, nodes, TypeError)
        case _:
            raise Unsupported(f"cannot call {describe(callee)}")


def method_call(name: str, arguments: Sequence[Item]) -> tuple[Node, list[Fault]]:
    if name == "total_seconds" and len(arguments) == 1 and isinstance(arguments[0], Duration):
        return invoke("__truediv__", [arguments[0].node, lit(MICROSECONDS_PER_SECOND)], TypeError)
    if name == "translate" and len(arguments) == 2:
        table = arguments[1]
        if not isinstance(table, Opaque):
            raise Unsupported("translate requires a constant dict")
        characters, replacements = translation_table(table.value)
        return invoke(
            ".translate",
            [node_of(arguments[0]), lit(characters), Lit(replacements, list_of(STR))],
            AttributeError,
        )
    if name in {"startswith", "endswith"}:
        matched = tuple_affixes(name, arguments)
        if matched is not None:
            return matched
    nodes = [
        iterable_node(argument) if name == "join" and index == 1 else node_of(argument)
        for index, argument in enumerate(arguments)
    ]
    if name == "strftime":
        return invoke(".strftime", nodes, AttributeError)
    folded = evaluated(nodes, lambda values: getattr(values[0], name)(*values[1:]))
    if folded is not None:
        return folded, []
    if name == "format":
        node, faults = format_template(nodes[0], nodes[1:])
        return fold(node), faults
    return invoke(f".{name}", nodes, AttributeError)


def sequence_call(name: str, item: Item) -> tuple[Node, list[Fault]]:
    if name in {"any", "all"} and isinstance(item, Generator):
        reduced = reduce_generator(name, item.node)
        if reduced is not None:
            return reduced
    if isinstance(item, Tuple) and name == "len":
        return lit(len(item.items)), []
    if isinstance(item, Set):
        if item.dynamic and name not in {"len", "bool"}:
            raise Unsupported("dynamic JSON sets support cardinality and predicates")
        return set_call(name, item.node)
    source = iterable_node(item)
    faults = nullable_faults(source)
    if name == "list" and source.type.kind is Kind.LIST:
        return source, faults
    if name in {"any", "all"} and source.type.item is not None and source.type.item != BOOL:
        if isinstance(source, Lit) and isinstance(source.value, tuple | list):
            require_bounded(len(source.value))
        element = Element("item", source.type.item)
        test, _ = truth(Value(element))
        source = Comprehension(source, element, test, TRUE, FALSE)
    if name == "sum":
        if isinstance(source, Lit) and source.value == ():
            source = Lit((), list_of(INT))
        if source.type.item is not None and source.type.item.kind is Kind.FLOAT:
            name = "sum_float"
    node, more_faults = invoke(name, [source], TypeError)
    return node, [*faults, *more_faults]


def set_call(name: str, source: Node) -> tuple[Node, list[Fault]]:
    operation = {
        "len": "set_size",
        "list": "set_list",
        "sorted": "sorted_set",
        "bool": "bool",
    }.get(name)
    if operation is None:
        raise Unsupported(f"{name} of a set")
    if (
        source.type.item is not None
        and source.type.item.kind is Kind.FLOAT
        and operation in {"set_size", "set_list", "sorted_set"}
    ):
        operation += "_float"
    return invoke(operation, [source], TypeError)
