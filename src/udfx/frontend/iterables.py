from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import reduce
from typing import cast

from udfx.errors import Unsupported
from udfx.frontend.constants import constant, constant_value, dictionary_method, scalar_map
from udfx.frontend.invocation import invoke, node_of
from udfx.frontend.sequence_calls import iterable_node
from udfx.frontend.sets import set_method
from udfx.frontend.values import (
    MAX_UNROLL,
    Generator,
    Item,
    Iterator,
    Method,
    Opaque,
    Set,
    Tuple,
    Value,
)
from udfx.ir import Element, ListValue, Lit, Node, lit
from udfx.ops import Fault, apply
from udfx.types import BOOL, FLOAT, INT, STR, Kind, list_of


def bounded(items: Sequence[Item]) -> Iterator:
    if len(items) > MAX_UNROLL:
        raise Unsupported(f"loops and comprehensions exceed {MAX_UNROLL} unrolled items")
    return Iterator(tuple(items))


def iterator_of(item: Item) -> tuple[Iterator, list[Fault]]:
    if isinstance(item, Iterator):
        return item, []
    if isinstance(item, Generator):
        return Iterator((), source=item.node), []
    if isinstance(item, Tuple):
        return bounded(item.items), []
    if isinstance(item, Value) and isinstance(item.node, ListValue):
        return bounded(tuple(Value(node) for node in item.node.items)), []
    if isinstance(item, Value) and not isinstance(item.node, Lit):
        source = item.node
        if source.type.kind is Kind.STR:
            source, faults = invoke("list", [source], TypeError)
            return Iterator((), source=source), faults
        if source.type.kind is not Kind.LIST:
            raise Unsupported("loops and comprehensions require a list")
        faults = [Fault(TypeError, apply("is_none", source))] if source.type.nullable else []
        return Iterator((), source=source), faults
    try:
        value = constant_value(item)
    except Unsupported:
        raise Unsupported("loops and comprehensions require a sequence") from None
    if not isinstance(value, tuple | list | range | dict | str):
        raise Unsupported("loops and comprehensions require a sequence")
    if len(value) > MAX_UNROLL:
        raise Unsupported(f"loops and comprehensions exceed {MAX_UNROLL} unrolled items")
    return bounded(tuple(constant(member) for member in value)), []


def source_of(iterator: Iterator) -> Node:
    if iterator.source is not None:
        return iterator.source
    nodes = tuple(node_of(item) for item in iterator.items)
    if nodes and any(node.type != nodes[0].type for node in nodes):
        raise Unsupported("comprehension items must share one type")
    result = list_of(nodes[0].type if nodes else STR)
    if all(isinstance(node, Lit) for node in nodes):
        return Lit(tuple(node.value for node in nodes if isinstance(node, Lit)), result)
    return ListValue(nodes, result)


def selected(items: tuple[Item, ...], index: Element) -> Item:
    if all(isinstance(item, Tuple) for item in items):
        tuples = [item.items for item in items if isinstance(item, Tuple)]
        return Tuple(tuple(selected(tuple(column), index) for column in zip(*tuples, strict=True)))
    return Value(apply("__getitem__", source_of(Iterator(items)), index))


def indexed(iterator: Iterator, name: str) -> tuple[Node, Element, Item]:
    if iterator.element is not None and iterator.value is not None:
        return source_of(iterator), iterator.element, iterator.value
    if iterator.source is None and any(isinstance(item, Tuple) for item in iterator.items):
        element = Element(name, INT)
        source = apply("range", lit(len(iterator.items)))
        return source, element, selected(iterator.items, element)
    source = source_of(iterator)
    if source.type.item is None:
        raise Unsupported("comprehension requires a list")
    element = Element(name, source.type.item)
    return source, element, Value(element)


def constant_range(arguments: Sequence[Item]) -> Opaque:
    values = [constant_value(argument) for argument in arguments]
    if not 1 <= len(values) <= 3 or not all(isinstance(value, int) for value in values):
        raise Unsupported("range requires one to three integer literals")
    sequence = range(*(value for value in values if isinstance(value, int)))
    if sequence and (sequence[-1] - sequence[0]) // sequence.step >= MAX_UNROLL:
        raise Unsupported(f"loops and comprehensions exceed {MAX_UNROLL} unrolled items")
    return Opaque(sequence)


def combined(name: str, arguments: Sequence[Item]) -> tuple[Iterator, list[Fault]]:
    start: Node = lit(0)
    faults: list[Fault] = []
    if name == "enumerate":
        if not 1 <= len(arguments) <= 2:
            raise Unsupported("enumerate requires an iterable and an optional start")
        if len(arguments) == 2:
            start = node_of(arguments[1])
            if isinstance(start, Lit) and isinstance(start.value, int):
                start = lit(int(start.value))
            if start.type.kind is not Kind.INT:
                raise Unsupported("enumerate start requires an integer")
            if start.type.nullable:
                faults.append(Fault(TypeError, apply("is_none", start)))
        arguments = arguments[:1]
    iterators: list[Iterator] = []
    for argument in arguments:
        iterator, pending = iterator_of(argument)
        if iterator.value is not None:
            raise Unsupported("nested paired iterators")
        iterators.append(iterator)
        faults.extend(pending)
    if all(iterator.source is None for iterator in iterators):
        items = tuple(
            Tuple(
                (Value(invoke("__add__", [start, lit(index)], TypeError)[0]), *members)
                if name == "enumerate"
                else members
            )
            for index, members in enumerate(
                zip(*(iterator.items for iterator in iterators), strict=False)
            )
        )
        return bounded(items), faults
    sources = [source_of(iterator) for iterator in iterators]
    length = reduce(
        lambda left, right: apply("min", left, right), (apply("len", source) for source in sources)
    )
    element = Element(f"{name}:index", INT)
    members = tuple(Value(apply("__getitem__", source, element)) for source in sources)
    value = Tuple(
        (Value(apply("__add__", element, start)), *members) if name == "enumerate" else members
    )
    return Iterator(
        (),
        source=apply("range", length),
        element=element,
        value=value,
        requires_native=any(iterator.requires_native for iterator in iterators),
    ), faults


def construct(
    callee: object, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> tuple[Item, list[Fault]] | None:
    if callee is dict:
        if len(arguments) > 1:
            raise Unsupported("dict requires at most one constant iterable")
        item = arguments[0] if arguments else Opaque(())
        value = (
            tuple(constant_value(member) for member in item.items)
            if isinstance(item, Iterator) and item.source is None
            else constant_value(item)
        )
        if not isinstance(value, dict | tuple | list):
            raise Unsupported("dict requires a constant iterable")
        entries = dict(cast("Mapping[object, object] | Iterable[tuple[object, object]]", value))
        entries.update((name, constant_value(item)) for name, item in keywords.items())
        return Opaque(scalar_map(entries.items())), []
    if keywords or (callee is not list and callee is not tuple):
        return None
    container = list if callee is list else tuple
    if not arguments:
        return Value(Lit((), list_of(STR)), container), []
    if len(arguments) != 1:
        return None
    item = arguments[0]
    if isinstance(item, Iterator | Tuple | Opaque) or (
        isinstance(item, Value) and item.container is tuple
    ):
        iterator, faults = iterator_of(item)
        if iterator.source is not None:
            raise Unsupported("dynamic iterators require a comprehension")
        return Value(source_of(iterator), container), faults
    if callee is tuple:
        source, faults = invoke("list", [node_of(item)], TypeError)
        return Value(source, tuple), faults
    return None


def range_iterator(arguments: Sequence[Item]) -> tuple[Item, list[Fault]]:
    if all(
        isinstance(argument, Value) and isinstance(argument.node, Lit) for argument in arguments
    ):
        return constant_range(arguments), []
    if not 1 <= len(arguments) <= 3:
        raise Unsupported("range requires one to three arguments")
    nodes = [node_of(argument) for argument in arguments]
    if len(nodes) > 1:
        if not isinstance(nodes[0], Lit) or (len(nodes) == 3 and not isinstance(nodes[2], Lit)):
            raise Unsupported("range start and step require integer literals")
        if len(nodes) == 2:
            nodes.append(lit(1))
    source, faults = invoke("range", nodes, TypeError)
    return Iterator((), source=source, requires_native=True), faults


def call_iterable(
    callee: Item, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> tuple[Item, list[Fault]] | None:
    if isinstance(callee, Method):
        if arguments and isinstance(arguments[0], Set):
            return set_method(callee.name, arguments, keywords)
        return dictionary_method(callee.name, arguments, keywords)
    if not isinstance(callee, Opaque):
        return None
    if callee.value is zip and keywords:
        if set(keywords) != {"strict"} or constant_value(keywords["strict"]) is not False:
            raise Unsupported("zip strict mode")
        keywords = {}
    if callee.value is enumerate and set(keywords) == {"start"} and len(arguments) == 1:
        arguments = [*arguments, keywords["start"]]
        keywords = {}
    if not keywords:
        if callee.value is range:
            return range_iterator(arguments)
        if callee.value is enumerate or callee.value is zip:
            return combined("enumerate" if callee.value is enumerate else "zip", arguments)
        if callee.value is set and len(arguments) == 1:
            item = arguments[0]
            if isinstance(item, Set):
                return item, []
            source = iterable_node(item)
            if source.type.kind is not Kind.STR and source.type.item not in (STR, INT, FLOAT, BOOL):
                raise Unsupported("set requires text or a scalar list")
            node, faults = invoke("list", [source], TypeError)
            return Set(node), faults
    return construct(callee.value, arguments, keywords)
