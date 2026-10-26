from __future__ import annotations

import operator
from collections import Counter
from functools import partial

from udfx.ir import Node, lit
from udfx.ops.core import is_none
from udfx.ops.registry import Fault, Op, apply, register
from udfx.ops.strings import out_of_range, slice_of
from udfx.semantics.lists import (
    SET_OPERATIONS,
    SET_PREDICATES,
    range_list,
    set_list,
    set_operation,
    set_size,
    sorted_set,
)
from udfx.types import BOOL, FLOAT, INT, STR, list_of


def frequency_order(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(counts, key=counts.__getitem__, reverse=True)


register(Op("frequency_order", (list_of(STR),), list_of(STR), frequency_order))

for item in (STR, INT, FLOAT, BOOL):
    sequence = list_of(item)
    register(Op("list", (sequence,), sequence, list))
    register(Op("__add__", (sequence, sequence), sequence, operator.add))
    register(Op("__mul__", (sequence, INT), sequence, operator.mul))
    register(Op("__mul__", (INT, sequence), sequence, operator.mul))
    if item != STR:
        register(Op("slice", (sequence, INT.or_none(), INT.or_none()), sequence, slice_of))
        register(Op("len", (sequence,), INT, len))
        register(Op("bool", (sequence.or_none(),), BOOL, bool))
        register(Op("is_none", (sequence.or_none(),), BOOL, is_none))
        if item != INT:
            register(Op("__contains__", (sequence, item), BOOL, operator.contains))
        register(Op("__getitem__", (sequence, INT), item, operator.getitem, out_of_range))

register(Op("__contains__", (list_of(INT.or_none()), INT.or_none()), BOOL, operator.contains))
register(Op("any", (list_of(BOOL),), BOOL, any))
register(Op("all", (list_of(BOOL),), BOOL, all))
register(Op("sum", (list_of(INT),), INT, sum))
register(Op("sum", (list_of(BOOL),), INT, sum))
register(Op("sum_float", (list_of(FLOAT),), FLOAT, sum))
register(Op("sorted", (list_of(STR),), list_of(STR), sorted))
for item in (STR, INT, FLOAT, BOOL):
    suffix = "_float" if item == FLOAT else ""
    register(Op(f"set_size{suffix}", (list_of(item),), INT, set_size))
    register(Op(f"set_list{suffix}", (list_of(item),), list_of(item), set_list))
    register(Op(f"sorted_set{suffix}", (list_of(item),), list_of(item), sorted_set))

for operation in (*SET_OPERATIONS, *SET_PREDICATES):
    register(
        Op(
            f"set_{operation}",
            (list_of(STR), list_of(STR)),
            BOOL if operation in SET_PREDICATES else list_of(STR),
            partial(set_operation, operation=operation),
        )
    )

for operation in SET_OPERATIONS:
    register(
        Op(
            f"set_{operation}_size",
            (list_of(STR), list_of(STR)),
            INT,
            partial(set_operation, operation=operation, size=True),
        )
    )


def range_faults(start: Node, stop: Node, step: Node) -> tuple[Fault, ...]:
    return (Fault(ValueError, apply("__eq__", step, lit(0))),)


register(Op("list", (STR,), list_of(STR), list))
register(Op("range", (INT,), list_of(INT), range_list))
register(Op("range", (INT, INT, INT), list_of(INT), range_list, range_faults))
