from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import cast

from udfx.frontend.calls import invoke
from udfx.frontend.constants import constant_value
from udfx.frontend.values import Duration, Item, Opaque, Value
from udfx.ir import TRUE, Node, lit
from udfx.ops import Fault
from udfx.semantics.datetimes import (
    MAXIMUM_SHIFT,
    MICROSECONDS_PER_DAY,
    MICROSECONDS_PER_SECOND,
    SECONDS_PER_DAY,
    duration_microseconds,
)
from udfx.types import Kind


def call_datetime(
    callee: Item, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> tuple[Item, list[Fault]] | None:
    if not isinstance(callee, Opaque) or callee.value is not timedelta:
        return None
    values = [constant_value(argument) for argument in arguments]
    named = {name: constant_value(value) for name, value in keywords.items()}
    try:
        duration = cast("Callable[..., timedelta]", timedelta)(*values, **named)
    except (TypeError, ValueError, OverflowError) as error:
        return Opaque(timedelta()), [Fault(type(error), TRUE)]
    return Opaque(duration), []


def datetime_binary(symbol: str, left: Item, right: Item) -> tuple[Item, list[Fault]] | None:
    if symbol == "+" and isinstance(left, Opaque) and isinstance(left.value, timedelta):
        left, right = right, left
    if not isinstance(left, Value) or left.node.type.kind is not Kind.DATETIME:
        return None
    if symbol == "-" and isinstance(right, Value) and right.node.type.kind is Kind.DATETIME:
        node, faults = invoke("datetime_difference", [left.node, right.node], TypeError)
        return Duration(node), faults
    if symbol in {"+", "-"} and isinstance(right, Opaque) and isinstance(right.value, timedelta):
        offset = duration_microseconds(right.value) * (-1 if symbol == "-" else 1)
        offset = max(-MAXIMUM_SHIFT, min(MAXIMUM_SHIFT, offset))
        node, faults = invoke("datetime_shift", [left.node, lit(offset)], TypeError)
        return Value(node), faults
    return None


def duration_attribute(node: Node, name: str) -> tuple[Node, list[Fault]]:
    divisor = MICROSECONDS_PER_DAY if name == "days" else MICROSECONDS_PER_SECOND
    result, faults = invoke("__floordiv__", [node, lit(divisor)], AttributeError)
    if name == "seconds":
        result, pending = invoke("__mod__", [result, lit(SECONDS_PER_DAY)], TypeError)
        faults.extend(pending)
    return result, faults
