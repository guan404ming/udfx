from __future__ import annotations

import operator

from udfx.ops.core import is_none
from udfx.ops.registry import Op, register
from udfx.ops.strings import out_of_range
from udfx.semantics.lists import set_list, set_size, sorted_set
from udfx.types import BOOL, FLOAT, INT, STR, list_of

for item in (STR, INT, FLOAT, BOOL):
    sequence = list_of(item)
    register(Op("list", (sequence,), sequence, list))
    register(Op("__add__", (sequence, sequence), sequence, operator.add))
    if item != STR:
        register(Op("len", (sequence,), INT, len))
        register(Op("bool", (sequence.or_none(),), BOOL, bool))
        register(Op("is_none", (sequence.or_none(),), BOOL, is_none))
        register(Op("__contains__", (sequence, item), BOOL, operator.contains))
        register(Op("__getitem__", (sequence, INT), item, operator.getitem, out_of_range))

register(Op("any", (list_of(BOOL),), BOOL, any))
register(Op("all", (list_of(BOOL),), BOOL, all))
register(Op("sum", (list_of(INT),), INT, sum))
register(Op("sum", (list_of(BOOL),), INT, sum))
register(Op("sum_float", (list_of(FLOAT),), FLOAT, sum))
register(Op("sorted", (list_of(STR),), list_of(STR), sorted))
register(Op("set_size", (list_of(STR),), INT, set_size))
register(Op("set_list", (list_of(STR),), list_of(STR), set_list))
register(Op("sorted_set", (list_of(STR),), list_of(STR), sorted_set))
