from __future__ import annotations

from udfx.errors import Unsupported
from udfx.frontend.invocation import invoke, node_of
from udfx.frontend.logic import negate
from udfx.frontend.reductions import reduce_generator
from udfx.frontend.values import (
    MAX_UNROLL,
    Generator,
    Item,
    Match,
    Opaque,
    Set,
    Tuple,
    Value,
    describe,
)
from udfx.ir import FALSE, TRUE, Comprehension, Element, Lit, Node, lit
from udfx.ops import Fault, apply
from udfx.types import BOOL, INT, Kind, list_of


def iterable_node(item: Item) -> Node:
    if isinstance(item, Generator):
        return item.node
    if isinstance(item, Set):
        return sequence_call("list", item)[0]
    if isinstance(item, Value) and item.container in (tuple, list):
        return item.node
    return node_of(item)


def sequence_call(name: str, item: Item) -> tuple[Node, list[Fault]]:
    if name in {"any", "all"} and isinstance(item, Generator):
        reduced = reduce_generator(name, item.node)
        if reduced is not None:
            return reduced, []
    if isinstance(item, Tuple) and name == "len":
        return lit(len(item.items)), []
    if isinstance(item, Set):
        operation = {
            "len": "set_size",
            "list": "set_list",
            "sorted": "sorted_set",
            "bool": "bool",
        }.get(name)
        if operation is None:
            raise Unsupported(f"{name} of a set")
        if (
            item.node.type.item is not None
            and item.node.type.item.kind is Kind.FLOAT
            and operation in {"set_size", "set_list", "sorted_set"}
        ):
            operation += "_float"
        return invoke(operation, [item.node], TypeError)
    source = iterable_node(item)
    faults = [Fault(TypeError, apply("is_none", source))] if source.type.nullable else []
    if name == "list" and source.type.kind is Kind.LIST:
        return source, faults
    if name in {"any", "all"} and source.type.item is not None and source.type.item != BOOL:
        if (
            isinstance(source, Lit)
            and isinstance(source.value, tuple | list)
            and len(source.value) > MAX_UNROLL
        ):
            raise Unsupported(f"loops and comprehensions exceed {MAX_UNROLL} unrolled items")
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


def truth(item: Item) -> tuple[Node, list[Fault]]:
    match item:
        case Tuple(items=items):
            return lit(bool(items)), []
        case Match():
            return negate(none_test(item)), []
        case Set(node=node):
            return invoke("bool", [node], TypeError)
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
