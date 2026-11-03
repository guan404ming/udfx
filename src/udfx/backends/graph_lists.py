from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from udfx.backends.graph import lower_block
from udfx.backends.indexing import every
from udfx.backends.rules import Lowered, Scope
from udfx.ir import FALSE, TRUE, Comprehension, Node
from udfx.semantics.assumptions import assuming
from udfx.semantics.facts import element_predicate

if TYPE_CHECKING:
    from udfx.backends.base import Backend


def lower_graph_comprehension[E](
    backend: Backend[E], node: Comprehension, scope: Scope[E]
) -> Lowered[E]:
    source = backend.lower(node.source, scope)
    if node.fault == FALSE:
        plain = plain_comprehension(backend, node, scope, source)
        if plain is not None:
            return plain
    guarded = False

    def finish(parts: Sequence[Lowered[E]]) -> Lowered[E]:
        condition, value, fault = parts
        exact = backend.exact(value)
        if node.condition != TRUE and exact is not None:
            exact = backend.select_exact(condition.expr, exact, None)
        return Lowered(
            backend.record([condition.expr, value.expr, fault.expr]),
            backend.conjunction(backend.exact(condition), exact, backend.exact(fault)),
        )

    def body(element: E) -> E:
        nonlocal guarded
        with assuming(element_predicate(node.source, node.element), True):
            output = lower_block(
                backend,
                (node.condition, node.value, node.fault),
                {**scope, node.element: element},
                finish,
            )
        guarded = output.exact is not None
        return output.consume(
            lambda part: backend.record([part.expr, backend.exact_expression(part)])
        )

    records = backend.list_transform(source.expr, body)

    def result(records: E) -> E:
        selected = (
            records
            if node.condition == TRUE
            else backend.list_filter(
                records, lambda value: backend.field(backend.field(value, 0), 0)
            )
        )
        values = backend.list_transform(
            selected, lambda value: backend.field(backend.field(value, 0), 1)
        )
        exact = (
            every(backend, backend.list_transform(records, lambda value: backend.field(value, 1)))
            if guarded
            else backend.literal(TRUE)
        )
        safe = (
            every(
                backend,
                backend.list_transform(
                    records, lambda value: backend.negate(backend.field(backend.field(value, 0), 2))
                ),
            )
            if node.fault != FALSE
            else backend.literal(TRUE)
        )
        return backend.record([values, exact, backend.negate(safe)])

    def fields(value: E) -> Lowered[E]:
        return Lowered(
            backend.field(value, 0),
            backend.conjunction(
                backend.exact(source), backend.field(value, 1) if guarded else None
            ),
            backend.field(value, 2) if node.fault != FALSE else None,
        )

    value = backend.bind(records, result)
    lowered = fields(value)
    if lowered.exact is None and lowered.overflow is None:
        return lowered
    return replace(
        lowered, prepare=lambda finish: backend.bind(value, lambda bound: finish(fields(bound)))
    )


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
