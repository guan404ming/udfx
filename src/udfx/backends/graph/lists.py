from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast

from udfx.backends.graph.bindings import lower_block
from udfx.backends.graph.siblings import SIBLINGS, share
from udfx.backends.rewrites.indexing import every
from udfx.backends.rules import Lowered, Scope
from udfx.graphs import unique_size
from udfx.ir import FALSE, TRUE, Comprehension, Node
from udfx.semantics.facts import element_predicate
from udfx.semantics.numbers.assumptions import assuming
from udfx.types import INT, STR

if TYPE_CHECKING:
    from udfx.backends.base import Backend

DISTINCT_SIZE = 100


def lower_graph_comprehension[E](
    backend: Backend[E], node: Comprehension, scope: Scope[E]
) -> Lowered[E]:
    sets, results = SIBLINGS.get() or ({}, {})
    members = sets.get(id(node), (node,))
    if len(members) == 1 and node.fault == FALSE:
        plain = plain_comprehension(backend, node, scope, backend.lower(node.source, scope))
        if plain is not None:
            return plain
    if id(node) not in results:
        results.update(zip(map(id, members), lower_members(backend, members, scope), strict=True))
    return cast("Lowered[E]", results[id(node)])


def lower_members[E](
    backend: Backend[E], nodes: Sequence[Comprehension], scope: Scope[E]
) -> list[Lowered[E]]:
    first = nodes[0]
    source = backend.lower(first.source, scope)
    roots = [part for node in nodes for part in (node.condition, node.value, node.fault)]
    guarded = False

    def finish(parts: Sequence[Lowered[E]]) -> Lowered[E]:
        records: list[E] = []
        exact: list[E | Literal[False] | None] = []
        for index, node in enumerate(nodes):
            condition, value, fault = parts[3 * index : 3 * index + 3]
            checked = backend.exact(value)
            if node.condition != TRUE and checked is not None:
                checked = backend.select_exact(condition.expr, checked, None)
            records.append(backend.record([condition.expr, value.expr, fault.expr]))
            exact.extend((backend.exact(condition), checked, backend.exact(fault)))
        return Lowered(backend.record(records), backend.conjunction(*exact))

    def body(element: E) -> E:
        nonlocal guarded
        with assuming(element_predicate(first.source, first.element), True):
            output = lower_block(backend, roots, {**scope, first.element: element}, finish)
        guarded = output.exact is not None
        return output.consume(
            lambda part: backend.record([part.expr, backend.exact_expression(part)])
        )

    distinct = (
        backend.list_distinct(source.expr, body)
        if first.element.type in (STR, INT) and unique_size(roots) > DISTINCT_SIZE
        else None
    )
    records = backend.list_transform(source.expr, body) if distinct is None else distinct

    def summary(records: E, index: int, node: Comprehension) -> list[E]:
        def part(value: E, field: int) -> E:
            return backend.field(backend.field(backend.field(value, 0), index), field)

        selected = (
            records
            if node.condition == TRUE
            else backend.list_filter(records, lambda value: part(value, 0))
        )
        safe = (
            every(
                backend,
                backend.list_transform(records, lambda value: backend.negate(part(value, 2))),
            )
            if node.fault != FALSE
            else backend.literal(TRUE)
        )
        return [
            backend.list_transform(selected, lambda value: part(value, 1)),
            backend.negate(safe),
        ]

    def result(records: E) -> E:
        exact = (
            every(backend, backend.list_transform(records, lambda value: backend.field(value, 1)))
            if guarded
            else backend.literal(TRUE)
        )
        return backend.record(
            [
                exact,
                *(
                    part
                    for index, node in enumerate(nodes)
                    for part in summary(records, index, node)
                ),
            ]
        )

    def fields(value: E, index: int) -> Lowered[E]:
        return Lowered(
            backend.field(value, 1 + 2 * index),
            backend.conjunction(
                backend.exact(source), backend.field(value, 0) if guarded else None
            ),
            backend.field(value, 2 + 2 * index) if nodes[index].fault != FALSE else None,
        )

    value = backend.bind(records, result)

    def member(index: int) -> Lowered[E]:
        lowered = fields(value, index)
        if lowered.exact is not None or lowered.overflow is not None:
            lowered = replace(
                lowered,
                prepare=lambda finish: backend.bind(
                    value, lambda bound: finish(fields(bound, index))
                ),
            )
        if backend.exact(source) is not None:
            return lowered
        return share(lowered, value, lambda bound: fields(bound, index))

    return [member(index) for index in range(len(nodes))]


def plain_comprehension[E](
    backend: Backend[E], node: Comprehension, scope: Scope[E], source: Lowered[E]
) -> Lowered[E] | None:
    guarded = False

    def part(root: Node, element: E) -> E:
        nonlocal guarded
        with assuming(element_predicate(node.source, node.element), True):
            output = lower_block(
                backend, (root,), {**scope, node.element: element}, lambda parts: parts[0]
            )
        guarded = guarded or any(
            value is not None for value in (output.exact, output.overflow, output.prepare)
        )
        return output.expr

    selected = (
        source.expr
        if node.condition == TRUE
        else backend.list_filter(source.expr, lambda element: part(node.condition, element))
    )
    values = backend.list_transform(selected, lambda element: part(node.value, element))
    return None if guarded else Lowered(values, backend.exact(source))
