from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from udfx.backends.graph import lower_block
from udfx.backends.indexing import every
from udfx.backends.rules import Lowered, Scope
from udfx.ir import FALSE, TRUE, Comprehension

if TYPE_CHECKING:
    from udfx.backends.base import Backend


def lower_graph_comprehension[E](
    backend: Backend[E], node: Comprehension, scope: Scope[E]
) -> Lowered[E]:
    source = backend.lower(node.source, scope)
    guarded = False

    def finish(parts: Sequence[Lowered[E]]) -> Lowered[E]:
        condition, value, fault = parts
        exact = backend.exact(value)
        if node.condition != TRUE and exact is not None:
            exact = backend.select(condition.expr, exact, backend.literal(TRUE))
        return Lowered(
            backend.record([condition.expr, value.expr, fault.expr]),
            backend.conjunction(backend.exact(condition), exact, backend.exact(fault)),
        )

    def body(element: E) -> E:
        nonlocal guarded
        output = lower_block(
            backend,
            (node.condition, node.value, node.fault),
            {**scope, node.element: element},
            finish,
        )
        guarded = output.exact is not None
        return backend.record([output.expr, backend.exact_expression(output)])

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

    value = backend.bind(records, result)
    return Lowered(
        backend.field(value, 0),
        backend.conjunction(backend.exact(source), backend.field(value, 1) if guarded else None),
        backend.field(value, 2) if node.fault != FALSE else None,
    )
