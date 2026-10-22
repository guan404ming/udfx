from dataclasses import replace

from udfx.frontend.logic import fold, negate
from udfx.frontend.values import MAX_UNROLL
from udfx.ir import FALSE, TRUE, Call, Comprehension, Element, Guarded, If, ListValue, Lit, Node
from udfx.ops.registry import apply
from udfx.types import BOOL


def select(condition: Node, then: Node, orelse: Node) -> Node:
    if condition == TRUE:
        return then
    if condition == FALSE or then == orelse:
        return orelse
    return If(condition, then, orelse, then.type)


def short_circuit(name: str, value: Node, remaining: Node) -> Node:
    decisive = TRUE if name == "any" else FALSE
    identity = FALSE if name == "any" else TRUE
    if value == decisive:
        return decisive
    if value == identity:
        return remaining
    if remaining == identity:
        return value
    then, orelse = (decisive, remaining) if name == "any" else (remaining, decisive)
    return If(value, then, orelse, BOOL)


def substitute(node: Node, element: Element, value: Node) -> Node:
    if node == element:
        return value
    if isinstance(node, Call):
        return fold(
            replace(
                node, arguments=tuple(substitute(part, element, value) for part in node.arguments)
            )
        )
    if isinstance(node, If):
        return select(
            substitute(node.condition, element, value),
            substitute(node.then, element, value),
            substitute(node.orelse, element, value),
        )
    if isinstance(node, Guarded):
        return Guarded(
            substitute(node.value, element, value), substitute(node.exact, element, value)
        )
    if isinstance(node, Comprehension):
        source = substitute(node.source, element, value)
        if node.element == element:
            return replace(node, source=source)
        return replace(
            node,
            source=source,
            value=substitute(node.value, element, value),
            condition=substitute(node.condition, element, value),
            fault=substitute(node.fault, element, value),
        )
    if isinstance(node, ListValue):
        return replace(node, items=tuple(substitute(part, element, value) for part in node.items))
    return node


def reduce_generator(name: str, node: Node) -> Node | None:
    if not isinstance(node, Comprehension) or not isinstance(node.source, Lit):
        return None
    items = node.source.value
    if not isinstance(items, tuple | list) or len(items) > MAX_UNROLL:
        return None
    result: Node = TRUE if name == "all" else FALSE
    for item in reversed(items):
        element = Lit(item, node.element.type)
        condition = substitute(node.condition, node.element, element)
        fault = substitute(node.fault, node.element, element)
        value = substitute(node.value, node.element, element)
        if value.type != BOOL:
            value = fold(apply("bool", value))
        value = select(condition, value, TRUE if name == "all" else FALSE)
        result = short_circuit(name, value, result)
        if fault != FALSE:
            result = Guarded(result, negate(fault))
    return result
