from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from udfx.semantics.numbers import FLOAT_INT_MAX

SET_ORDER_MAX_SIZE = 1
NATIVE_LIST_LIMIT = 10_000
SET_OPERATIONS = ("intersection", "union", "difference", "symmetric_difference")
SET_PREDICATES = ("issubset", "issuperset", "isdisjoint")


def matches_itself(value: object) -> bool:
    return operator.eq(value, value)


def set_size(values: list[Any]) -> int:
    return len(set(values))


def set_list(values: list[Any]) -> list[Any]:
    return list(set(values))


def sorted_set(values: list[Any]) -> list[Any]:
    return sorted(set(values))


def set_operation(
    left: list[str], right: list[str], operation: str, *, size: bool = False
) -> list[str] | bool | int:
    result = getattr(set(left), operation)(set(right))
    return (len(result) if size else sorted(result)) if isinstance(result, set) else result


def range_list(*arguments: int) -> list[int]:
    return list(range(*arguments))


@dataclass(frozen=True)
class SumPrimitives:
    literal: Callable[[object], Any]
    when: Callable[[Any, Any, Any], Any]
    length: Callable[[Any], Any]
    minimum: Callable[[Any], Any]
    maximum: Callable[[Any], Any]
    coalesce: Callable[[Any, Any], Any]


def integer_sum_exact(primitives: SumPrimitives, value: Any) -> Any:
    literal = primitives.literal
    length = primitives.length(value)
    bound = literal(FLOAT_INT_MAX) / primitives.when(length > literal(0), length, literal(1))
    return primitives.coalesce(
        (primitives.minimum(value) >= literal(0) - bound) & (primitives.maximum(value) <= bound),
        literal(True),
    )


@dataclass(frozen=True)
class ListPrimitives:
    literal: Callable[[object], Any]
    when: Callable[[Any, Any, Any], Any]
    length: Callable[[Any], Any]
    repeat: Callable[[Any, Any], Any]
    range: Callable[[Any, Any, Any], Any]


def repeat_list(primitives: ListPrimitives, value: Any, count: Any) -> tuple[Any, Any]:
    literal = primitives.literal
    length = primitives.length(value)
    empty = length == literal(0)
    capacity = literal(NATIVE_LIST_LIMIT) / primitives.when(empty, literal(1), length)
    exact = empty | (count <= capacity)
    safe = primitives.when(exact & ~empty & (count > literal(0)), count, literal(0))
    return primitives.repeat(value, safe), exact


def native_range(primitives: ListPrimitives, start: Any, stop: Any, step: Any) -> tuple[Any, Any]:
    literal = primitives.literal
    when = primitives.when
    limit = literal(NATIVE_LIST_LIMIT)
    zero = literal(0)
    exact = (
        (start >= literal(-NATIVE_LIST_LIMIT))
        & (start <= limit)
        & (stop >= literal(-NATIVE_LIST_LIMIT))
        & (stop <= limit)
    )
    exact = exact & (step >= literal(-NATIVE_LIST_LIMIT)) & (step <= limit)
    start = when(exact, start, zero)
    stop = when(exact, stop, zero)
    step = when(exact & (step != zero), step, literal(1))
    forward = step > zero
    stop = when((forward & (stop > start)) | (~forward & (stop < start)), stop, start)
    return primitives.range(start, stop, step), exact


@dataclass(frozen=True)
class SetPrimitives:
    literal: Callable[[object], Any]
    length: Callable[[Any], Any]
    distinct: Callable[[Any], Any]
    sort: Callable[[Any], Any]
    contains: Callable[[Any, Any], Any]
    intersection: Callable[[Any, Any], Any]
    union: Callable[[Any, Any], Any]
    concatenate: Callable[[Any, Any], Any]
    difference: Callable[[Any, Any], Any]
    size: Callable[[Any], Any] | None = None
    symmetric_difference: Callable[[Any, Any], Any] | None = None


def native_set_operation(
    primitives: SetPrimitives, left: Any, right: Any, operation: str, *, size: bool = False
) -> Any:
    match operation:
        case "intersection" | "isdisjoint":
            result = primitives.intersection(left, right)
        case "union":
            result = primitives.union(left, right)
        case "difference" | "issubset":
            result = primitives.difference(left, right)
        case "issuperset":
            result = primitives.difference(right, left)
        case "symmetric_difference" if primitives.symmetric_difference is not None:
            result = primitives.symmetric_difference(left, right)
        case _:
            result = primitives.concatenate(
                primitives.difference(left, right), primitives.difference(right, left)
            )
    if operation in SET_PREDICATES:
        return primitives.length(result) == primitives.literal(0)
    return primitives.length(result) if size else primitives.sort(result)


def native_set(
    primitives: SetPrimitives, value: Any, operation: str, floating: bool
) -> tuple[Any, Any]:
    exact = None
    if operation == "set_size":
        result = (
            primitives.length(primitives.distinct(value))
            if primitives.size is None
            else primitives.size(value)
        )
    else:
        result = primitives.distinct(value)
        if operation == "sorted_set":
            result = primitives.sort(result)
        else:
            exact = primitives.length(result) <= primitives.literal(SET_ORDER_MAX_SIZE)
    if floating:
        exact = primitives.literal(True) if exact is None else exact
        for item in (float("nan"), 0.0, -0.0):
            exact = exact & ~primitives.contains(value, primitives.literal(item))
    return result, exact
