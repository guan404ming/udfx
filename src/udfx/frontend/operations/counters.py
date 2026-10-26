from __future__ import annotations

import collections
import heapq
from collections.abc import Mapping, Sequence

from udfx.errors import Unsupported
from udfx.frontend.core.expressions import invoke, node_of
from udfx.frontend.core.values import Counter, CounterKey, Item, Opaque, Value
from udfx.frontend.operations.builtins import iterable_node
from udfx.ir import lit
from udfx.ops import Fault
from udfx.types import INT, STR, list_of


def call_counter(
    callee: Item, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> tuple[Item, list[Fault]] | None:
    if not isinstance(callee, Opaque):
        return None
    if callee.value is collections.Counter:
        if keywords or len(arguments) != 1:
            raise Unsupported("Counter requires one string iterable")
        source = iterable_node(arguments[0])
        source, faults = invoke("list", [source], TypeError)
        if source.type != list_of(STR):
            raise Unsupported("Counter requires strings")
        return Counter(source), faults
    if callee.value is not heapq.nlargest:
        return None
    if len(arguments) != 2 or set(keywords) != {"key"}:
        raise Unsupported("nlargest requires a Counter and its get method as key")
    count, counter = arguments
    key = keywords["key"]
    if not isinstance(counter, Counter) or not isinstance(key, CounterKey) or key.owner != counter:
        raise Unsupported("nlargest requires a Counter and its get method as key")
    ordered, faults = invoke("frequency_order", [counter.node], TypeError)
    limit, pending = invoke("max", [lit(0), node_of(count)], TypeError)
    value, slicing = invoke("slice", [ordered, lit(None, INT.or_none()), limit], TypeError)
    return Value(value), [*faults, *pending, *slicing]
