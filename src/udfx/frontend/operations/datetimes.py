from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta, timezone
from types import BuiltinMethodType
from typing import cast

from udfx.frontend.core.expressions import invoke, none_test
from udfx.frontend.core.logic import conjunction, negate
from udfx.frontend.core.values import Duration, Item, Opaque, Value
from udfx.frontend.resolve.constants import constant, constant_value
from udfx.ir import FALSE, TRUE, Node, lit
from udfx.ops import Fault
from udfx.semantics.temporal.datetimes import (
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
    if not isinstance(callee, Opaque):
        return None
    function = callee.value
    calendar = function is date or (
        isinstance(function, BuiltinMethodType)
        and function.__self__ is date
        and function.__name__ in {"fromisoformat", "fromordinal"}
    )
    if function is not timedelta and function is not timezone and not calendar:
        return None
    values = [constant_value(argument) for argument in arguments]
    named = {name: constant_value(value) for name, value in keywords.items()}
    try:
        value = cast("Callable[..., object]", function)(*values, **named)
    except (TypeError, ValueError, OverflowError) as error:
        return constant(date.min if calendar else timedelta()), [Fault(type(error), TRUE)]
    return constant(value), []


def datetime_binary(symbol: str, left: Item, right: Item) -> tuple[Item, list[Fault]] | None:
    if symbol == "+" and isinstance(left, Opaque) and isinstance(left.value, timedelta):
        left, right = right, left
    if not isinstance(left, Value) or left.node.type.kind not in {Kind.DATE, Kind.DATETIME}:
        return None
    prefix = left.node.type.kind.value
    if (
        symbol == "-"
        and isinstance(right, Value)
        and right.node.type.kind in {Kind.DATE, Kind.DATETIME}
    ):
        if left.node.type.kind is not right.node.type.kind:
            return Duration(lit(0)), [Fault(TypeError, TRUE)]
        node, faults = invoke(f"{prefix}_difference", [left.node, right.node], TypeError)
        return Duration(node), faults
    if symbol in {"+", "-"} and isinstance(right, Opaque) and isinstance(right.value, timedelta):
        offset = (
            right.value.days
            if left.node.type.kind is Kind.DATE
            else duration_microseconds(right.value)
        ) * (-1 if symbol == "-" else 1)
        if left.node.type.kind is Kind.DATETIME:
            offset = max(-MAXIMUM_SHIFT, min(MAXIMUM_SHIFT, offset))
        node, faults = invoke(f"{prefix}_shift", [left.node, lit(offset)], TypeError)
        return Value(node), faults
    return None


def datetime_comparison(symbol: str, left: Item, right: Item) -> tuple[Node, list[Fault]] | None:
    if not (
        isinstance(left, Value)
        and isinstance(right, Value)
        and {left.node.type.kind, right.node.type.kind} == {Kind.DATE, Kind.DATETIME}
    ):
        return None
    if symbol in {"==", "!="}:
        equal = conjunction(none_test(left), none_test(right))
        return negate(equal) if symbol == "!=" else equal, []
    return FALSE, [Fault(TypeError, TRUE)]


def duration_attribute(node: Node, name: str) -> tuple[Node, list[Fault]]:
    divisor = MICROSECONDS_PER_DAY if name == "days" else MICROSECONDS_PER_SECOND
    result, faults = invoke("__floordiv__", [node, lit(divisor)], AttributeError)
    if name == "seconds":
        result, pending = invoke("__mod__", [result, lit(SECONDS_PER_DAY)], TypeError)
        faults.extend(pending)
    return result, faults
