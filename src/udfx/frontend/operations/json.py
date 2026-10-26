import inspect
import json
from collections.abc import Mapping, Sequence

from udfx.errors import Unsupported
from udfx.frontend.core.expressions import invoke, node_of
from udfx.frontend.core.json import array, iterable_faults, packed, string, strings, type_fault
from udfx.frontend.core.values import Item, Json, Method, Opaque, Set, Value
from udfx.frontend.operations.builtins import method_call
from udfx.frontend.resolve.constants import constant, constant_value
from udfx.frontend.resolve.functions import bind
from udfx.ir import FALSE, TRUE, Comprehension, Element, If, Lit, Node, lit
from udfx.ops import Fault, apply
from udfx.semantics.json import attribute_types
from udfx.types import INT, STR, Kind


def serialized(value: Node, ensure_ascii: bool) -> Node:
    if value.type.kind is Kind.NONE:
        return lit("null")
    if value.type.kind is Kind.STR:
        return apply("json.dumps", value, lit(ensure_ascii))
    if value.type.kind is not Kind.LIST or value.type.item is None:
        raise Unsupported("json.dumps supports strings and lists of strings")
    element = Element("json item", value.type.item)
    items = Comprehension(value, element, serialized(element, ensure_ascii), TRUE, FALSE)
    joined = apply(".join", lit(", "), items)
    result = apply("__add__", apply("__add__", lit("["), joined), lit("]"))
    return If(apply("is_none", value), lit("null"), result, STR) if value.type.nullable else result


def call_json(
    callee: Item, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> tuple[Item, list[Fault]] | None:
    if isinstance(callee, Method) and arguments and isinstance(arguments[0], Json):
        if keywords or callee.name.startswith("__") or not hasattr(str, callee.name):
            raise Unsupported(f"JSON method {callee.name}")
        source = arguments[0].node
        value, faults = method_call(callee.name, [Value(string(source)), *arguments[1:]])
        return Value(value), [
            type_fault(source, AttributeError, attribute_types(callee.name)),
            *faults,
        ]
    if not isinstance(callee, Opaque):
        return None
    if callee.value is json.loads:
        return loaded(arguments, keywords)
    if len(arguments) == 1 and isinstance(arguments[0], Json) and not keywords:
        found = consumed(callee.value, arguments[0])
        if found is not None:
            return found
    if callee.value is not json.dumps:
        return None
    return dumped(arguments, keywords)


def dumped(arguments: Sequence[Item], keywords: Mapping[str, Item]) -> tuple[Item, list[Fault]]:
    signature = inspect.signature(json.dumps)
    bindings = bind(signature, arguments, keywords)
    ensure_ascii = True
    for name, item in bindings.items():
        if name == "obj":
            continue
        option = constant_value(item)
        if name == "ensure_ascii" and isinstance(option, bool):
            ensure_ascii = option
        elif option != signature.parameters[name].default:
            raise Unsupported(f"json.dumps option {name} is unsupported")
    item = bindings["obj"]
    if isinstance(item, Json):
        source = item.node
        result = If(
            apply("json.is_type", source, lit("str")),
            serialized(string(source), ensure_ascii),
            serialized(strings(source), ensure_ascii),
            STR,
        )
    else:
        result = serialized(node_of(item), ensure_ascii)
    return Value(result), []


def loaded(arguments: Sequence[Item], keywords: Mapping[str, Item]) -> tuple[Item, list[Fault]]:
    signature = inspect.signature(json.loads)
    bindings = bind(signature, arguments, keywords)
    for name, item in bindings.items():
        if name != "s" and constant_value(item) != signature.parameters[name].default:
            raise Unsupported(f"json.loads option {name} is unsupported")
    source = node_of(bindings["s"])
    if isinstance(source, Lit):
        if not isinstance(source.value, str):
            return Value(lit(None)), [Fault(TypeError, TRUE)]
        try:
            return constant(json.loads(source.value)), []
        except (ValueError, RecursionError) as error:
            return Value(lit(None)), [Fault(type(error), TRUE)]
    valid, faults = invoke("json.valid", [source], TypeError)
    return Json(source), [*faults, Fault(json.JSONDecodeError, apply("not", valid))]


def consumed(callee: object, item: Json) -> tuple[Item, list[Fault]] | None:
    source = item.node
    faults = iterable_faults(source)
    if callee is len:
        value = If(
            apply("json.is_type", source, lit("str")),
            apply("len", string(source)),
            apply("len", array(source)),
            INT,
        )
        return Value(value), faults
    if callee is sorted:
        return Json(serialized(apply("sorted", strings(source)), False), list), faults
    if callee is list or callee is tuple:
        value = If(
            apply("json.is_type", source, lit("str")),
            serialized(apply("list", string(source)), False),
            packed(array(source)),
            STR,
        )
        return Json(value, list if callee is list else tuple), faults
    if callee is set:
        return Set(strings(source), dynamic=True), faults
    return None
