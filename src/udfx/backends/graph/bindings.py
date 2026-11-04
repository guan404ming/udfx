from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from udfx.backends.graph.siblings import View, fusing, share, shared, sharing, siblings
from udfx.backends.rules import Lowered, Scope
from udfx.graphs import free_finder, needed, substitute
from udfx.ir import FALSE, Call, Col, Comprehension, Element, Lit, Node, children
from udfx.semantics.bindings import bound_sources
from udfx.types import Kind

if TYPE_CHECKING:
    from udfx.backends.base import Backend

type Place[E] = tuple[int, bool | None, bool] | tuple[int, View[E]]


def dependencies(node: Node, find: Callable[[Node], frozenset[Element]]) -> tuple[Node, ...]:
    if not isinstance(node, Comprehension):
        return children(node)
    outer = find(node)
    found: dict[int, Node] = {}
    seen: set[int] = set()

    def visit(part: Node) -> None:
        if id(part) in seen or isinstance(part, Element | Lit):
            return
        seen.add(id(part))
        if find(part) <= outer:
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
    parts: dict[int, tuple[Node, ...]] = {}
    find = free_finder()

    bound = {id(node) for node in incoming}

    def visit(node: Node) -> None:
        counts[id(node)] += 1
        if id(node) in seen or id(node) in bound:
            return
        seen.add(id(node))
        parts[id(node)] = dependencies(node, find)
        for child in parts[id(node)]:
            visit(child)
        order.append(node)

    for root in roots:
        visit(root)
    fused = siblings(order)
    forced: dict[int, int] = {}
    while True:
        groups: dict[int, list[Node]] = {}
        depths: dict[int, int] = {}
        grouped: set[int] = set()
        for node in order:
            level = max((depths.get(id(part), 0) for part in parts[id(node)]), default=0)
            level = max(level, forced.get(id(node), 0))
            if id(node) in fused or (
                counts[id(node)] > 1
                and not isinstance(node, Col | Element | Lit)
                and not recomputed(node, grouped)
            ):
                grouped.add(id(node))
                groups.setdefault(level, []).append(node)
                level += 1
            depths[id(node)] = level
        found = {
            key: max(depths[id(member)] - 1 for member in members) for key, members in fused.items()
        }
        if found == forced:
            return order, list(groups.values())
        forced = found


def recomputed(node: Node, grouped: set[int]) -> bool:
    return (
        isinstance(node, Call)
        and node.type.kind in (Kind.INT, Kind.BOOL)
        and not node.op.costly
        and all(
            isinstance(part, Col | Element | Lit) or id(part) in grouped for part in node.arguments
        )
    )


def lower_graph[E](backend: Backend[E], node: Node, scope: Scope[E]) -> Lowered[E]:
    return lower_block(backend, (node,), scope, lambda values: values[0])


def lower_block[E](
    backend: Backend[E],
    roots: Sequence[Node],
    incoming: Scope[E],
    finish: Callable[[Sequence[Lowered[E]]], Lowered[E]],
) -> Lowered[E]:
    with sharing():
        return lower_layers(backend, roots, incoming, finish)


def lower_layers[E](
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

    def lower(nodes: Sequence[Node], local: dict[int, Lowered[E]]) -> list[Lowered[E]]:
        replacements = {key: symbols[key] for key in local}
        substituted = [substitute(node, replacements) for node in nodes]
        bound = {symbols[key]: value for key, value in local.items()}
        with bound_sources(sources), fusing(substituted):
            return [backend.lower(node, bound) for node in substituted]

    def exact_state(value: Lowered[E]) -> bool | None:
        return None if value.exact is None else value.exact is not False

    def pack(local: dict[int, Lowered[E]]) -> tuple[E, dict[int, Place[E]]]:
        values: list[E] = []
        offsets: dict[int, Place[E]] = {}
        bases: dict[int, int] = {}
        for key, value in local.items():
            if (found := shared(value)) is not None:
                if id(found[0]) not in bases:
                    bases[id(found[0])] = len(values)
                    values.append(found[0])
                offsets[key] = bases[id(found[0])], found[1]
                continue
            offsets[key] = len(values), exact_state(value), value.overflow is not None
            values.append(value.expr)
            if value.exact is not None and value.exact is not False:
                values.append(backend.predicate(value.exact))
            if value.overflow is not None:
                values.append(value.overflow)
        return backend.record(values), offsets

    def unpack(record: E, offsets: dict[int, Place[E]]) -> dict[int, Lowered[E]]:
        return {key: place(record, value) for key, value in offsets.items()}

    def place(record: E, found: Place[E]) -> Lowered[E]:
        if len(found) == 2:
            base = backend.field(record, found[0])
            return share(found[1](base), base, found[1])
        offset, exact, overflow = found
        return Lowered(
            backend.field(record, offset),
            False if exact is False else backend.field(record, offset + 1) if exact else None,
            backend.field(record, offset + 1 + (exact is True)) if overflow else None,
        )

    def advance(index: int, local: dict[int, Lowered[E]]) -> tuple[E, dict[int, Place[E]]]:
        variables = {
            **local,
            **dict(zip(map(id, groups[index]), lower(groups[index], local), strict=True)),
        }
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
        lowered = finish(lower(roots, local))
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
