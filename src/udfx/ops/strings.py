from __future__ import annotations

import operator

from udfx.ir import Node, lit
from udfx.ops.registry import Fault, Op, apply, register
from udfx.types import BOOL, INT, STR, list_of

STRINGS = list_of(STR)


def empty_separator(_: Node, separator: Node) -> tuple[Fault, ...]:
    return (Fault(ValueError, apply("not", apply("bool", separator))),)


def out_of_range(container: Node, index: Node) -> tuple[Fault, ...]:
    length = apply("len", container)
    too_high = apply("not", apply("__gt__", length, index))
    too_low = apply("__gt__", lit(0), apply("__add__", index, length))
    return (Fault(IndexError, apply("or", too_high, too_low)),)


def slice_of(value: str, start: int | None, stop: int | None) -> str:
    return value[start:stop]


register(Op(".lower", (STR,), STR, str.lower))
register(Op(".upper", (STR,), STR, str.upper))
register(Op(".strip", (STR,), STR, str.strip))
register(Op(".lstrip", (STR,), STR, str.lstrip))
register(Op(".rstrip", (STR,), STR, str.rstrip))
register(Op(".split", (STR,), STRINGS, str.split))
register(Op(".split", (STR, STR), STRINGS, str.split, empty_separator))
register(Op(".find", (STR, STR), INT, str.find))
register(Op(".rfind", (STR, STR), INT, str.rfind))
register(Op(".count", (STR, STR), INT, str.count))
register(Op(".startswith", (STR, STR), BOOL, str.startswith))
register(Op(".endswith", (STR, STR), BOOL, str.endswith))
register(Op(".replace", (STR, STR, STR), STR, str.replace))
register(Op(".join", (STR, STRINGS), STR, str.join))
register(Op("__add__", (STR, STR), STR, operator.add))
register(Op("__eq__", (STR.or_none(), STR.or_none()), BOOL, operator.eq))
register(Op("__ne__", (STR.or_none(), STR.or_none()), BOOL, operator.ne))
register(Op("__contains__", (STR, STR), BOOL, operator.contains))
register(Op("__contains__", (STRINGS, STR), BOOL, operator.contains))
register(Op("len", (STR,), INT, len))
register(Op("len", (STRINGS,), INT, len))
register(Op("__getitem__", (STR, INT), STR, operator.getitem, out_of_range))
register(Op("__getitem__", (STRINGS, INT), STR, operator.getitem, out_of_range))
register(Op("slice", (STR, INT.or_none(), INT.or_none()), STR, slice_of))
