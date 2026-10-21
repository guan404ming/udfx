from __future__ import annotations

import operator

from udfx.ir import Node, lit
from udfx.ops.registry import Fault, Op, apply, register
from udfx.semantics.text import CASE_METHODS, PADDING_METHODS, PREDICATES
from udfx.types import BOOL, INT, STR, list_of

STRINGS = list_of(STR)
OPTIONAL_STRINGS = list_of(STR.or_none())


def empty_separator(_: Node, separator: Node, *limits: Node) -> tuple[Fault, ...]:
    return (Fault(ValueError, apply("__eq__", separator, lit(""))),)


def none_item(_: Node, items: Node) -> tuple[Fault, ...]:
    return (Fault(TypeError, apply("__contains__", items, lit(None, STR.or_none()))),)


def out_of_range(container: Node, index: Node) -> tuple[Fault, ...]:
    length = apply("len", container)
    too_high = apply("not", apply("__gt__", length, index))
    too_low = apply("__gt__", lit(0), apply("__add__", index, length))
    return (Fault(IndexError, apply("or", too_high, too_low)),)


def slice_of[Sequence: (str, list[str], list[str | None])](
    value: Sequence, start: int | None, stop: int | None, step: int | None = None
) -> Sequence:
    return value[start:stop:step]


def zero_step(_: Node, start: Node, stop: Node, step: Node) -> tuple[Fault, ...]:
    return (Fault(ValueError, apply("__eq__", step, lit(0))),)


def invalid_fill(_: Node, width: Node, fill: Node) -> tuple[Fault, ...]:
    return (Fault(TypeError, apply("__ne__", apply("len", fill), lit(1))),)


for name in CASE_METHODS:
    register(Op(f".{name}", (STR,), STR, getattr(str, name)))
for name in PADDING_METHODS:
    register(Op(f".{name}", (STR, INT), STR, getattr(str, name)))
    if name != "zfill":
        register(Op(f".{name}", (STR, INT, STR), STR, getattr(str, name), invalid_fill))
for name in PREDICATES:
    register(Op(f".{name}", (STR,), BOOL, getattr(str, name)))
for name in ("strip", "lstrip", "rstrip"):
    register(Op(f".{name}", (STR,), STR, getattr(str, name)))
    register(Op(f".{name}", (STR, STR.or_none()), STR, getattr(str, name)))
for name, method in ((".split", str.split), (".rsplit", str.rsplit)):
    register(Op(name, (STR,), STRINGS, method))
    register(Op(name, (STR, STR.or_none()), STRINGS, method, empty_separator))
    register(Op(name, (STR, STR.or_none(), INT), STRINGS, method, empty_separator))
register(Op(".find", (STR, STR), INT, str.find))
register(Op(".rfind", (STR, STR), INT, str.rfind))
register(Op(".count", (STR, STR), INT, str.count))
register(Op(".startswith", (STR, STR), BOOL, str.startswith))
register(Op(".endswith", (STR, STR), BOOL, str.endswith))
register(Op(".replace", (STR, STR, STR), STR, str.replace))
register(Op(".join", (STR, STRINGS), STR, str.join))
register(Op(".join", (STR, OPTIONAL_STRINGS), STR, str.join, none_item))
register(Op(".join", (STR, STR), STR, str.join))
register(Op("__add__", (STR, STR), STR, operator.add))
register(Op("__eq__", (STR.or_none(), STR.or_none()), BOOL, operator.eq))
register(Op("__ne__", (STR.or_none(), STR.or_none()), BOOL, operator.ne))
register(Op("__contains__", (STR, STR), BOOL, operator.contains))
register(Op("__contains__", (OPTIONAL_STRINGS, STR.or_none()), BOOL, operator.contains))
register(Op("len", (STR,), INT, len))
register(Op("len", (OPTIONAL_STRINGS,), INT, len))
register(Op("__getitem__", (STR, INT), STR, operator.getitem, out_of_range))
register(Op("__getitem__", (STRINGS, INT), STR, operator.getitem, out_of_range))
register(Op("__getitem__", (OPTIONAL_STRINGS, INT), STR.or_none(), operator.getitem, out_of_range))
for sequence in (STR, STRINGS, OPTIONAL_STRINGS):
    register(Op("slice", (sequence, INT.or_none(), INT.or_none()), sequence, slice_of))
    register(
        Op(
            "slice",
            (sequence, INT.or_none(), INT.or_none(), INT.or_none()),
            sequence,
            slice_of,
            zero_step,
        )
    )
