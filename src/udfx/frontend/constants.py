from __future__ import annotations

from collections.abc import Iterable

from udfx.errors import Unsupported
from udfx.frontend.calls import invoke
from udfx.frontend.logic import choose, conj, neg
from udfx.frontend.values import Item, Opaque, Value
from udfx.ir import TRUE, Lit, Node, lit
from udfx.ops import Fault
from udfx.types import list_of

SCALARS = (str, int, float, bool)

ITEMS = (str, int)


def is_scalar(value: object) -> bool:
    return value is None or type(value) in SCALARS


def constant(value: object) -> Item:
    if is_scalar(value):
        return Value(lit(value))
    if isinstance(value, tuple | list | set | frozenset) and value:
        kinds = {type(item) for item in value}
        if len(kinds) == 1 and kinds <= set(ITEMS):
            items = tuple(sorted(value) if isinstance(value, set | frozenset) else value)
            item_type = lit(items[0]).type
            return Value(Lit(items, list_of(item_type)))
    return Opaque(value)


def literal(item: Item) -> object:
    match item:
        case Value(node=Lit(value=value)):
            return value
        case Opaque(obj=obj):
            return obj
        case _:
            raise Unsupported(f"expected a constant, got {item}")


def scalar_map(pairs: Iterable[tuple[object, object]]) -> dict[object, object]:
    entries: dict[object, object] = {}
    for key, value in pairs:
        if not (is_scalar(key) and is_scalar(value)):
            raise Unsupported(f"dict entry {key!r}: {value!r} is not a scalar constant")
        entries[key] = value
    return entries


def lookup(mapping: object, key: Node) -> tuple[Node, list[Fault]]:
    if not isinstance(mapping, dict):
        raise Unsupported(f"subscript of {mapping!r}")
    entries = scalar_map(mapping.items())
    if isinstance(key, Lit) or not entries:
        if isinstance(key, Lit) and key.value in entries:
            return lit(entries[key.value]), []
        return lit(None), [Fault(KeyError, TRUE)]
    cases: list[tuple[Node, Node]] = []
    faults: list[Fault] = []
    miss: Node = TRUE
    for entry, value in entries.items():
        test, test_faults = invoke("__eq__", [key, lit(entry)], KeyError)
        cases.append((test, lit(value)))
        faults.extend(test_faults)
        miss = conj(miss, neg(test))
    return choose(cases), [*faults, Fault(KeyError, miss)]
