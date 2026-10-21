from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import cast

from udfx.errors import Unsupported
from udfx.frontend.binding import bind
from udfx.frontend.formatting import format_template
from udfx.frontend.logic import disjunction, fold, negate
from udfx.frontend.values import (
    MAX_UNROLL,
    Generator,
    Item,
    Match,
    Method,
    Opaque,
    Set,
    Value,
    describe,
    named,
)
from udfx.ir import FALSE, TRUE, Call, Comprehension, Element, Lit, Node, lit
from udfx.ops import Fault, Op, apply, matching
from udfx.semantics.datetimes import format_parts, parse_parts
from udfx.types import BOOL, INT, Kind, list_of, type_of

CALLABLES: tuple[tuple[object, str], ...] = (
    (abs, "abs"),
    (any, "any"),
    (all, "all"),
    (bool, "bool"),
    (float, "float"),
    (int, "int"),
    (len, "len"),
    (list, "list"),
    (set, "set"),
    (sorted, "sorted"),
    (sum, "sum"),
    (max, "max"),
    (min, "min"),
    (round, "round"),
    (str, "str"),
    (math.ceil, "math.ceil"),
    (math.floor, "math.floor"),
    (math.isinf, "math.isinf"),
    (math.isnan, "math.isnan"),
    (math.log10, "math.log10"),
    (math.sqrt, "math.sqrt"),
    (re.compile, "re.compile"),
    (re.sub, "re.sub"),
    (re.search, "re.search"),
    (re.match, "re.match"),
    (re.fullmatch, "re.fullmatch"),
    (re.findall, "re.findall"),
    (re.split, "re.split"),
    (re.Pattern.sub, "re.sub"),
    (re.Pattern.search, "re.search"),
    (re.Pattern.match, "re.match"),
    (re.Pattern.fullmatch, "re.fullmatch"),
    (re.Pattern.findall, "re.findall"),
    (re.Pattern.split, "re.split"),
)

CALLABLE_IDENTITIES = {
    ("builtins", "pandas", "isna"): "pandas.isna",
    ("builtins", "pandas", "notna"): "pandas.notna",
    ("builtins", "pandas.core.dtypes.missing", "isna"): "pandas.isna",
    ("builtins", "pandas.core.dtypes.missing", "notna"): "pandas.notna",
    ("numpy", "ufunc", "isnan"): "numpy.isnan",
    ("numpy", "ufunc", "log"): "numpy.log",
    ("numpy", "ufunc", "sqrt"): "numpy.sqrt",
    ("numpy", "ufunc", "exp"): "numpy.exp",
    ("numpy", "ufunc", "absolute"): "numpy.abs",
    ("numpy", "ufunc", "floor"): "numpy.floor",
    ("numpy", "ufunc", "ceil"): "numpy.ceil",
    ("numpy", "numpy", "round"): "numpy.round",
}


def node_of(item: Item) -> Node:
    if not isinstance(item, Value):
        raise Unsupported(f"expected a value, got {describe(item)}")
    if item.container in (tuple, set, frozenset):
        raise Unsupported(f"constant container {describe(item)} outside an in test")
    return item.node


def iterable_node(item: Item) -> Node:
    if isinstance(item, Generator):
        return item.node
    if isinstance(item, Set):
        return invoke("set_list", [item.node], TypeError)[0]
    if isinstance(item, Value) and item.container in (tuple, list):
        return item.node
    return node_of(item)


def known_op(callee: object) -> str | None:
    if (
        getattr(callee, "__self__", None) is datetime
        and getattr(callee, "__name__", "") == "strptime"
    ):
        return "datetime.strptime"
    if isinstance(getattr(callee, "__self__", None), re.Pattern):
        callee = getattr(re.Pattern, getattr(callee, "__name__", ""), callee)
    for known, name in CALLABLES:
        if known is callee:
            return name
    owner = type(callee)
    module = getattr(callee, "__module__", "")
    name = getattr(callee, "__qualname__", None) or getattr(callee, "__name__", "")
    if owner.__module__ == "numpy" and owner.__name__ == "ufunc":
        module = "ufunc"
    return CALLABLE_IDENTITIES.get((owner.__module__, module, name))


def op_name(callee: object) -> str:
    found = known_op(callee)
    if found is None:
        raise Unsupported(f"unknown callable {named(callee)}")
    return found


def overload(name: str, arguments: Sequence[Node]) -> tuple[Op, list[Node]]:
    types = [argument.type for argument in arguments]
    candidates = matching(name, types)
    others = {type.kind for type in types if type.kind is not Kind.NONE}
    preferred = [
        op
        for op in candidates
        if all(
            parameter.kind in others
            for type, parameter in zip(types, op.parameters, strict=True)
            if type.kind is Kind.NONE
        )
    ]
    chosen = preferred or candidates
    if not chosen:
        raise Unsupported(f"no op {name}({', '.join(str(type.strict()) for type in types)})")
    op = chosen[0]
    typed = [
        Lit(None, parameter) if argument.type.kind is Kind.NONE else argument
        for argument, parameter in zip(arguments, op.parameters, strict=True)
    ]
    return op, typed


def invoke(
    name: str, arguments: Sequence[Node], first_null_kind: type[Exception]
) -> tuple[Node, list[Fault]]:
    if (
        name in {".center", ".ljust", ".rjust"}
        and len(arguments) == 3
        and not isinstance(arguments[2], Lit)
    ):
        raise Unsupported("fill character must be a literal")
    if name in {".strftime", "datetime.strptime"}:
        if (
            len(arguments) != 2
            or not isinstance(arguments[1], Lit)
            or not isinstance(arguments[1].value, str)
        ):
            raise Unsupported("datetime format must be a string literal")
        if name == "datetime.strptime":
            parse_parts(arguments[1].value)
        else:
            format_parts(arguments[1].value)
    op, arguments = overload(name, arguments)
    call = Call(op, tuple(arguments))
    folded = fold(call)
    if isinstance(folded, Lit) and folded.type.kind is not Kind.LIST:
        return folded, []
    faults = [
        Fault(first_null_kind if index == 0 else TypeError, apply("is_none", argument))
        for index, (argument, parameter) in enumerate(zip(arguments, op.parameters, strict=True))
        if argument.type.nullable and not parameter.nullable
    ]
    faults.extend(op.faults(*arguments))
    return (call if isinstance(folded, Lit) else folded), faults


def evaluated(nodes: Sequence[Node], run: Callable[[list[object]], object]) -> Lit | None:
    values = [
        list(node.value)
        if node.type.kind is Kind.LIST and isinstance(node.value, tuple)
        else node.value
        for node in nodes
        if isinstance(node, Lit)
    ]
    if len(values) < len(nodes):
        return None
    try:
        value = run(values)
    except Exception:
        return None
    type = type_of(value)
    return None if type is None else Lit(value, type)


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
        if isinstance(arguments[0], Set) or name in {"list", "sorted", "any", "all", "sum"}:
            return sequence_call(name, arguments[0])
    match callee:
        case Method(name=name):
            return method_call(name, arguments)
        case Opaque(value=value):
            nodes = [node_of(argument) for argument in arguments]
            name = op_name(value)
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
    if isinstance(item, Set):
        operation = {
            "len": "set_size",
            "list": "set_list",
            "sorted": "sorted_set",
            "bool": "bool",
        }.get(name)
        if operation is None:
            raise Unsupported(f"{name} of a set")
        return invoke(operation, [item.node], TypeError)
    source = iterable_node(item)
    faults = [Fault(TypeError, apply("is_none", source))] if source.type.nullable else []
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
