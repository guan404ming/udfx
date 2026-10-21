from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import reduce
from types import FunctionType, GenericAlias, UnionType

from udfx.errors import Unsupported
from udfx.frontend.calls import invoke, none_test
from udfx.frontend.logic import choose, conjunction, negate
from udfx.frontend.values import Function, Item, Opaque, RuntimeType, Value, describe, named
from udfx.ir import TRUE, Col, Lit, Node, lit
from udfx.ops import Fault
from udfx.semantics.scalars import PYTHON_TYPES
from udfx.types import list_of

SCALARS = (str, int, float, bool)

ITEMS = (str, int)

ANNOTATIONS = (type, UnionType, GenericAlias)

GENERICS = (list, dict, tuple, set, frozenset, type)


def runtime_type(item: Item) -> Item:
    if isinstance(item, Value) and isinstance(item.node, Col):
        return RuntimeType(item.node)
    return Opaque(type(constant_value(item)))


def type_test(item: Item, expected: object, subclasses: bool = False) -> Node:
    if isinstance(expected, tuple) and subclasses:
        tests = [type_test(item, member, True) for member in expected]
        return negate(reduce(conjunction, map(negate, tests), TRUE))
    if not isinstance(expected, type) or expected not in PYTHON_TYPES.values():
        raise Unsupported(f"type test against {named(expected)}")
    if not isinstance(item, RuntimeType):
        actual = constant_value(item)
        if not isinstance(actual, type):
            raise Unsupported(f"type test of {named(actual)}")
        return lit(issubclass(actual, expected) if subclasses else actual is expected)
    actual = PYTHON_TYPES[item.node.type.kind]
    matches = issubclass(actual, expected) if subclasses else actual is expected
    if not item.node.type.nullable or matches == (expected is type(None)):
        return lit(matches)
    missing = none_test(Value(item.node))
    return negate(missing) if matches else missing


def type_comparison(left: Item, right: Item) -> Node | None:
    if isinstance(left, RuntimeType):
        return type_test(left, constant_value(right))
    if isinstance(right, RuntimeType):
        return type_test(right, constant_value(left))
    if isinstance(left, Opaque) and isinstance(left.value, type):
        return type_test(left, constant_value(right))
    return None


def is_scalar(value: object) -> bool:
    return value is None or type(value) in SCALARS


def constant(value: object) -> Item:
    if isinstance(value, FunctionType):
        return Function(value.__code__, (), value)
    if is_scalar(value):
        return Value(lit(value))
    if isinstance(value, tuple | list | set | frozenset):
        present = [item for item in value if item is not None]
        kinds = {type(item) for item in present}
        if len(kinds) == 1 and kinds <= set(ITEMS):
            missing = len(value) - len(present)
            ordered = isinstance(value, set | frozenset)
            items = (*sorted(present), *[None] * missing) if ordered else tuple(value)
            item_type = lit(present[0]).type
            return Value(
                Lit(items, list_of(item_type.or_none() if missing else item_type)), type(value)
            )
    return Opaque(value)


def constant_value(item: Item) -> object:
    match item:
        case Value(node=Lit(value=value)):
            return value
        case Opaque(value=value):
            return value
        case _:
            raise Unsupported(f"expected a constant, got {describe(item)}")


def is_annotation(value: object) -> bool:
    if isinstance(value, tuple):
        return all(is_annotation(member) for member in value)
    return value is None or type(value) in ANNOTATIONS


def annotation(operation: Callable[..., object], *items: Item) -> Opaque:
    values = [constant_value(item) for item in items]
    if not all(is_annotation(value) for value in values):
        raise Unsupported(f"static {operation.__name__} of {values!r}")
    try:
        return Opaque(operation(*values))
    except TypeError as error:
        raise Unsupported(f"static {operation.__name__} of {values!r}") from error


def scalar_map(pairs: Iterable[tuple[object, object]]) -> dict[object, object]:
    entries: dict[object, object] = {}
    for key, value in pairs:
        if not (is_scalar(key) and is_scalar(value)):
            raise Unsupported(f"dict entry {key!r}: {value!r} is not a scalar constant")
        entries[key] = value
    return entries


def lookup(mapping: object, key: Node, default: Node | None = None) -> tuple[Node, list[Fault]]:
    if not isinstance(mapping, dict):
        raise Unsupported(f"subscript of {named(mapping)}")
    entries = scalar_map(mapping.items())
    if isinstance(key, Lit) or not entries:
        if isinstance(key, Lit) and key.value in entries:
            return lit(entries[key.value]), []
        return (default, []) if default is not None else (lit(None), [Fault(KeyError, TRUE)])
    cases: list[tuple[Node, Node]] = []
    faults: list[Fault] = []
    miss: Node = TRUE
    for entry, value in entries.items():
        test, test_faults = invoke("__eq__", [key, lit(entry)], KeyError)
        cases.append((test, lit(value)))
        faults.extend(test_faults)
        miss = conjunction(miss, negate(test))
    if default is not None:
        return choose([*cases, (miss, default)]), faults
    return choose(cases), [*faults, Fault(KeyError, miss)]
