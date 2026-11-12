from __future__ import annotations

import itertools
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import duckdb
from duckdb import Expression

from udfx.backends.base import RESERVED
from udfx.backends.engines.duckdb.rules import (
    NESTED,
    bind,
    column,
    constant,
    function,
    lambda_expression,
)

type Body = Callable[[Expression], Expression]

ELEMENTS = itertools.count()


@dataclass
class Registry:
    subqueries: bool
    records: dict[int, tuple[Expression, Sequence[Expression]]] = field(default_factory=dict)
    columns: dict[int, tuple[Expression, str]] = field(default_factory=dict)
    sequences: dict[int, tuple[Expression, Expression, str]] = field(default_factory=dict)
    concatenations: dict[int, tuple[Expression, Sequence[Expression]]] = field(default_factory=dict)


REGISTRY: ContextVar[Registry | None] = ContextVar("udfx_duckdb_registry", default=None)


def element() -> str:
    return f"{RESERVED}element {next(ELEMENTS)}"


@contextmanager
def registry(subqueries: bool) -> Generator[None]:
    token = REGISTRY.set(Registry(subqueries))
    try:
        yield
    finally:
        REGISTRY.reset(token)


def active() -> Registry | None:
    state = REGISTRY.get()
    return None if state is None or not state.subqueries or NESTED.get() else state


def concatenate_strings(left: Expression, right: Expression) -> Expression:
    state = REGISTRY.get()
    if state is None:
        return function("concat", left, right)
    parts = [
        part
        for value in (left, right)
        for part in state.concatenations.get(id(value), (value, (value,)))[1]
    ]
    joined = function("concat", *parts)
    state.concatenations[id(joined)] = joined, parts
    return joined


def struct_record(values: Sequence[Expression]) -> Expression:
    packed = function(
        "struct_pack", *(value.alias(str(index)) for index, value in enumerate(values))
    )
    state = REGISTRY.get()
    if state is not None:
        state.records[id(packed)] = packed, values
    return packed


def struct_field(value: Expression, index: int) -> Expression:
    state = REGISTRY.get()
    found = None if state is None else state.columns.get(id(value))
    if found is None:
        return function("struct_extract", value, constant(str(index)))
    return column(f"{found[1]} {index}")


def project(state: Registry, value: Expression, last: bool) -> tuple[str, Expression]:
    name = element()
    placeholder = column(name)
    found = None if last else state.records.get(id(value))
    if found is None:
        return f"{value} AS {placeholder}", placeholder
    state.columns[id(placeholder)] = placeholder, name
    parts = (f"{item} AS {column(f'{name} {index}')}" for index, item in enumerate(found[1]))
    return ", ".join(parts), placeholder


def bind_sequence(value: Expression, steps: Sequence[Body]) -> Expression:
    state = active()
    if state is None:
        items = function("list_value", value)
        for step in steps:
            items = function("list_transform", items, lambda_expression(element(), step))
        return function("list_extract", items, constant(1))
    projection, current = project(state, value, not steps)
    query = f"SELECT {projection}"
    for index, step in enumerate(steps, 1):
        projection, current = project(state, step(current), index == len(steps))
        query = f"SELECT {projection} FROM ({query})"
    result = duckdb.SQLExpression(f"(SELECT {current} FROM ({query}))")
    state.sequences[id(result)] = result, current, query
    return result


def bind_value(value: Expression, body: Body) -> Expression:
    state = active()
    found = None if state is None else state.sequences.get(id(value))
    if found is None:
        return bind(value, body, element())
    return duckdb.SQLExpression(f"(SELECT {body(found[1])} FROM ({found[2]}))")
