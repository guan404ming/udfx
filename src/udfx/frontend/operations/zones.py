from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone

from udfx.errors import Unsupported
from udfx.frontend.core.expressions import invoke, node_of, nullable_faults
from udfx.frontend.core.values import Aware, Duration, Item, Method, Opaque, Value
from udfx.frontend.operations.builtins import call
from udfx.frontend.operations.datetimes import datetime_binary
from udfx.frontend.resolve.constants import constant_value
from udfx.ir import FALSE, TRUE, Lit, Node, lit
from udfx.ops import Fault
from udfx.semantics.datetimes import EPOCH, MICROSECONDS_PER_SECOND, duration_microseconds
from udfx.types import Kind

DELEGATED = frozenset({"date", "weekday", "isoweekday", "toordinal", "strftime"})
ZONE_DIRECTIVES = re.compile(r"(%%|%:z|%z|%Z)")


def offset(zone: timezone) -> int:
    return duration_microseconds(zone.utcoffset(None))


def naive(item: Item) -> bool:
    return isinstance(item, Value) and item.node.type.kind is Kind.DATETIME


def fixed_zone(item: Item) -> timezone | None:
    value = constant_value(item)
    if value is not None and not isinstance(value, timezone):
        raise Unsupported("timezone must be a datetime.timezone constant")
    return value


def shift(node: Node, microseconds: int) -> tuple[Node, list[Fault]]:
    if microseconds == 0:
        return node, []
    return invoke("datetime_shift", [node, lit(microseconds)], TypeError)


def rebased(item: Aware, zone: timezone) -> Aware | None:
    if not isinstance(item.node, Lit) or not isinstance(item.node.value, datetime):
        return None
    try:
        value = item.node.value.replace(tzinfo=item.zone).astimezone(zone)
    except OverflowError:
        return None
    return Aware(lit(value.replace(tzinfo=None)), zone)


def aligned(left: Aware, right: Aware) -> tuple[Aware, Aware]:
    if offset(left.zone) != offset(right.zone):
        moved = rebased(right, left.zone)
        if moved is not None:
            return left, moved
        moved = rebased(left, right.zone)
        if moved is not None:
            return moved, right
    return left, right


def instant_difference(left: Aware, right: Aware) -> tuple[Node, list[Fault]]:
    left, right = aligned(left, right)
    node, faults = invoke("datetime_difference", [left.node, right.node], TypeError)
    gap = offset(left.zone) - offset(right.zone)
    if gap == 0:
        return node, faults
    node, more = invoke("__sub__", [node, lit(gap)], TypeError)
    return node, [*faults, *more]


def aware_comparison(name: str, left: Item, right: Item) -> tuple[Node, list[Fault]] | None:
    if not isinstance(left, Aware) and not isinstance(right, Aware):
        return None
    if isinstance(left, Aware) and isinstance(right, Aware):
        left, right = aligned(left, right)
        if offset(left.zone) == offset(right.zone):
            return invoke(name, [left.node, right.node], TypeError)
        difference, faults = instant_difference(left, right)
        node, more = invoke(name, [difference, lit(0)], TypeError)
        return node, [*faults, *more]
    if name in {"__eq__", "__ne__"}:
        return lit(name == "__ne__"), []
    return FALSE, [Fault(TypeError, TRUE)]


def aware_binary(symbol: str, left: Item, right: Item) -> tuple[Item, list[Fault]] | None:
    if symbol == "+" and isinstance(right, Aware) and isinstance(left, Opaque):
        left, right = right, left
    if symbol not in {"+", "-"}:
        return None
    if not isinstance(left, Aware):
        if symbol == "-" and isinstance(right, Aware) and naive(left):
            return Duration(lit(0)), [Fault(TypeError, TRUE)]
        return None
    if isinstance(right, Opaque) and isinstance(right.value, timedelta):
        shifted = datetime_binary(symbol, Value(left.node), right)
        if shifted is None:
            return None
        item, faults = shifted
        return Aware(node_of(item), left.zone), faults
    if symbol == "-" and isinstance(right, Aware):
        node, faults = instant_difference(left, right)
        return Duration(node), faults
    if symbol == "-" and naive(right):
        return Duration(lit(0)), [Fault(TypeError, TRUE)]
    return None


def zone_format(format: Item, zone: timezone) -> Item:
    text = constant_value(format)
    if not isinstance(text, str):
        return format
    sample = datetime(2000, 1, 1, tzinfo=zone)
    parts = ZONE_DIRECTIVES.split(text)
    return Value(
        lit(
            "".join(
                sample.strftime(part).replace("%", "%%") if part in {"%z", "%:z", "%Z"} else part
                for part in parts
            )
        )
    )


def convert(owner: Aware, zone: timezone) -> tuple[Item, list[Fault]]:
    if zone is owner.zone:
        return owner, []
    universal, faults = shift(owner.node, -offset(owner.zone))
    node, more = shift(universal, offset(zone))
    return Aware(node, zone), [*faults, *more]


def timestamp(owner: Aware) -> tuple[Item, list[Fault]]:
    instant, faults = invoke("datetime_difference", [owner.node, lit(EPOCH)], TypeError)
    gap = offset(owner.zone)
    if gap:
        instant, more = invoke("__sub__", [instant, lit(gap)], TypeError)
        faults = [*faults, *more]
    node, more = invoke("__truediv__", [instant, lit(MICROSECONDS_PER_SECOND)], TypeError)
    return Value(node), [*faults, *more]


def replace_zone(
    owner: Item, rest: Sequence[Item], keywords: Mapping[str, Item]
) -> tuple[Item, list[Fault]]:
    if rest or set(keywords) != {"tzinfo"}:
        raise Unsupported("datetime replace supports only tzinfo")
    node = owner.node if isinstance(owner, Aware) else node_of(owner)
    faults = [] if isinstance(owner, Aware) else nullable_faults(node, AttributeError)
    zone = fixed_zone(keywords["tzinfo"])
    return (Value(node) if zone is None else Aware(node, zone)), faults


def aware_method(
    owner: Aware, name: str, rest: list[Item], keywords: Mapping[str, Item]
) -> tuple[Item, list[Fault]]:
    if name == "astimezone" and len(rest) + len(keywords) == 1 and set(keywords) <= {"tz"}:
        zone = fixed_zone(rest[0] if rest else keywords["tz"])
        if zone is not None:
            return convert(owner, zone)
    if keywords:
        raise Unsupported(f"keyword arguments to aware datetime {name}")
    if name in DELEGATED:
        if name == "strftime" and rest:
            rest[0] = zone_format(rest[0], owner.zone)
        node, faults = call(Method(name), [Value(owner.node), *rest])
        return Value(node), faults
    if rest:
        raise Unsupported(f"arguments to aware datetime {name}")
    match name:
        case "utcoffset":
            return Duration(lit(offset(owner.zone))), []
        case "tzname":
            return Value(lit(owner.zone.tzname(None))), []
        case "dst":
            return Value(lit(None)), []
        case "timestamp":
            return timestamp(owner)
        case _:
            raise Unsupported(f"aware datetime method {name}")


def call_zone(
    callee: Item, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> tuple[Item, list[Fault]] | None:
    if not isinstance(callee, Method) or not arguments:
        return None
    owner, rest = arguments[0], list(arguments[1:])
    if callee.name == "replace" and (isinstance(owner, Aware) or naive(owner)):
        return replace_zone(owner, rest, keywords)
    if isinstance(owner, Aware):
        return aware_method(owner, callee.name, rest, keywords)
    return None
