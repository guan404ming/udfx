from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Container, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from udfx.backends.rules import Lowered, Scope
from udfx.graphs import free, substitute
from udfx.ir import FALSE, Col, Comprehension, Element, Lit, Node, children
from udfx.semantics.bindings import bound_sources

if TYPE_CHECKING:
    from udfx.backends.base import Backend


def dependencies(node: Node) -> tuple[Node, ...]:
    if not isinstance(node, Comprehension):
        return children(node)
    outer = free(node)
    found: dict[int, Node] = {}

    def visit(part: Node) -> None:
        if id(part) in found or isinstance(part, Element | Lit):
            return
        if free(part) <= outer:
            found[id(part)] = part
            return
        for child in children(part):
            visit(child)

    for part in (node.value, node.condition, node.fault):
        visit(part)
    return (node.source, *found.values())


def layers(roots: Sequence[Node], incoming: Scope[object]) -> tuple[list[Node], list[list[Node]]]:
    order: list[Node] = []
    seen: set[int] = set()
    counts: Counter[int] = Counter()

    bound = {id(node) for node in incoming}

    def visit(node: Node) -> None:
        counts[id(node)] += 1
        if id(node) in seen or id(node) in bound:
            return
        seen.add(id(node))
        for child in dependencies(node):
            visit(child)
        order.append(node)

    for root in roots:
        visit(root)
    groups: dict[int, list[Node]] = {}
    depths: dict[int, int] = {}
    for node in order:
        level = max((depths.get(id(part), 0) for part in dependencies(node)), default=0)
        if counts[id(node)] > 1 and not isinstance(node, Col | Element | Lit):
            groups.setdefault(level, []).append(node)
            level += 1
        depths[id(node)] = level
    return order, list(groups.values())


def needed(roots: Sequence[Node], variables: Container[int]) -> set[int]:
    found: set[int] = set()
    seen: set[int] = set()

    def visit(node: Node) -> None:
        if id(node) in variables:
            found.add(id(node))
        elif id(node) not in seen:
            seen.add(id(node))
            for child in children(node):
                visit(child)

    for root in roots:
        visit(root)
    return found


def lower_graph[E](backend: Backend[E], node: Node, scope: Scope[E]) -> Lowered[E]:
    return lower_block(backend, (node,), scope, lambda values: values[0])


def lower_block[E](
    backend: Backend[E],
    roots: Sequence[Node],
    incoming: Scope[E],
    finish: Callable[[Sequence[Lowered[E]]], Lowered[E]],
) -> Lowered[E]:
    order, groups = layers(roots, incoming)
    if not groups:
        return finish([backend.lower(node, incoming) for node in roots])
    scope = {
        id(key): value if isinstance(value, Lowered) else Lowered(value)
        for key, value in incoming.items()
    }
    scope.update(
        (id(node), backend.lower(node, incoming))
        for node in order
        if isinstance(node, Col | Element) and id(node) not in scope
    )
    symbols: dict[int, Node] = {
        id(node): Element(f"udfx shared {id(roots)}:{index}", node.type)
        for index, node in enumerate(
            [
                *incoming,
                *(node for node in order if isinstance(node, Col | Element)),
                *(node for group in groups for node in group),
            ]
        )
    }
    metadata: list[tuple[bool | None, bool]] = []
    sources = {
        symbols[id(node)]: substitute(
            node, {key: symbol for key, symbol in symbols.items() if key != id(node)}
        )
        for node in (*incoming, *order)
        if id(node) in symbols
    }

    def lower(node: Node, local: dict[int, Lowered[E]]) -> Lowered[E]:
        replacements = {key: symbols[key] for key in local}
        with bound_sources(sources):
            return backend.lower(
                substitute(node, replacements),
                {symbols[key]: value for key, value in local.items()},
            )

    def exact_state(value: Lowered[E]) -> bool | None:
        return None if value.exact is None else value.exact is not False

    def pack(local: dict[int, Lowered[E]]) -> tuple[E, dict[int, tuple[int, bool | None, bool]]]:
        values: list[E] = []
        offsets: dict[int, tuple[int, bool | None, bool]] = {}
        for key, value in local.items():
            offsets[key] = len(values), exact_state(value), value.overflow is not None
            values.append(value.expr)
            if value.exact is not None and value.exact is not False:
                values.append(backend.predicate(value.exact))
            if value.overflow is not None:
                values.append(value.overflow)
        return backend.record(values), offsets

    def unpack(
        record: E, offsets: dict[int, tuple[int, bool | None, bool]]
    ) -> dict[int, Lowered[E]]:
        return {
            key: Lowered(
                backend.field(record, offset),
                False if exact is False else backend.field(record, offset + 1) if exact else None,
                backend.field(record, offset + 1 + (exact is True)) if overflow else None,
            )
            for key, (offset, exact, overflow) in offsets.items()
        }

    def advance(
        index: int, local: dict[int, Lowered[E]]
    ) -> tuple[E, dict[int, tuple[int, bool | None, bool]]]:
        variables = {**local, **{id(node): lower(node, local) for node in groups[index]}}
        required = needed(
            [*roots, *(node for group in groups[index + 1 :] for node in group)], variables
        )
        return pack({key: value for key, value in variables.items() if key in required})

    record, offsets = advance(0, scope)

    def step(index: int) -> Callable[[E], E]:
        def body(value: E) -> E:
            nonlocal offsets
            packed, offsets = advance(index, unpack(value, offsets))
            return packed

        return body

    def output(value: E) -> E:
        local = unpack(value, offsets)
        lowered = finish([lower(node, local) for node in roots])
        metadata.append((exact_state(lowered), lowered.overflow is not None))
        return backend.record(
            [
                lowered.expr,
                backend.predicate(lowered.exact),
                lowered.overflow if lowered.overflow is not None else backend.literal(FALSE),
            ]
        )

    result = backend.bind_sequence(
        record, [*(step(index) for index in range(1, len(groups))), output]
    )
    exact, overflow = metadata[0]

    def fields(value: E) -> Lowered[E]:
        return Lowered(
            backend.field(value, 0),
            False if exact is False else backend.field(value, 1) if exact else None,
            backend.field(value, 2) if overflow else None,
        )

    return replace(
        fields(result),
        prepare=(lambda finish: backend.bind(result, lambda value: finish(fields(value))))
        if exact is True or overflow
        else None,
    )
