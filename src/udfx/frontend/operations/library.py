from __future__ import annotations

from collections.abc import Mapping, Sequence

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import Evaluation, always, single
from udfx.frontend.core.values import Item, Opaque, Row, Tuple, Value
from udfx.frontend.operations.builtins import call
from udfx.frontend.operations.counters import call_counter
from udfx.frontend.operations.datetimes import call_datetime
from udfx.frontend.operations.iterables import call_iterable
from udfx.frontend.operations.json import call_json
from udfx.frontend.operations.paths import call_path
from udfx.frontend.operations.regex import call_regex
from udfx.frontend.resolve.constants import (
    constant,
    constant_value,
    make_translation_table,
    runtime_type,
    type_test,
)
from udfx.ir import Node


def reject_rows(arguments: Sequence[Item], keywords: Mapping[str, Item], where: str) -> None:
    if any(isinstance(argument, Row) for argument in (*arguments, *keywords.values())):
        raise Unsupported("passing a row to a call", where)


def require_empty_keywords(keywords: Item, where: str) -> None:
    if constant_value(keywords) != {}:
        raise Unsupported("nonempty unpacked keyword arguments", where)


def unpacked_arguments(packed: Item, where: str) -> tuple[Item, ...]:
    if isinstance(packed, Tuple):
        return packed.items
    members = constant_value(packed)
    if not isinstance(members, tuple):
        raise Unsupported("unpacked call arguments require a tuple", where)
    return tuple(constant(member) for member in members)


def ordinary_call(
    callee: Item, arguments: Sequence[Item], keywords: Mapping[str, Item], path: Node
) -> Evaluation:
    special = (
        call_datetime(callee, arguments, keywords)
        or call_counter(callee, arguments, keywords)
        or call_json(callee, arguments, keywords)
        or call_path(callee, arguments, keywords)
        or call_regex(callee, arguments, keywords, path)
    )
    if special is not None:
        item, faults = special
        return single(item, faults)
    try:
        iterable = call_iterable(callee, arguments, keywords)
    except (TypeError, ValueError) as error:
        return always(type(error))
    if iterable is not None:
        item, faults = iterable
        return single(item, faults)
    if isinstance(callee, Opaque) and not keywords:
        if callee.value is str.maketrans:
            item, faults = make_translation_table(arguments)
            return single(item, faults)
        if callee.value is type and len(arguments) == 1:
            return single(runtime_type(arguments[0]))
        if callee.value is isinstance and len(arguments) == 2:
            test = type_test(runtime_type(arguments[0]), constant_value(arguments[1]), True)
            return single(Value(test))
    node, faults = call(callee, arguments, keywords)
    return single(Value(node), faults)
