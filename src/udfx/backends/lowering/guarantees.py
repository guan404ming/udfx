from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from udfx.ir import FALSE, Call, If, Lit, Node

if TYPE_CHECKING:
    from udfx.backends.base import Backend
    from udfx.backends.rules import Lowered, Scope


@dataclass(frozen=True)
class Guarantee:
    predicate: Node
    default: Lit | None = None


def constant(node: Node) -> Lit | None:
    return (
        node
        if isinstance(node, Lit) and isinstance(node.value, str | int | bool | type(None))
        else None
    )


def required[E](
    values: Iterable[Lowered[E]], predicate: Node | None = None
) -> tuple[Guarantee, ...]:
    found = [] if predicate is None else [Guarantee(predicate)]
    for value in values:
        for guarantee in value.guarantees:
            if guarantee.default is None and guarantee not in found:
                found.append(guarantee)
    return tuple(found)


def checked[E](value: Lowered[E]) -> tuple[Guarantee, ...]:
    return tuple(
        Guarantee(guarantee.predicate)
        for guarantee in value.guarantees
        if guarantee.default is None or guarantee.default == FALSE
    )


def selects_default[E](value: Lowered[E], predicate: Node, default: Node) -> bool:
    literal = constant(default)
    return any(
        guarantee.predicate == predicate
        and (guarantee.default is None or (literal is not None and guarantee.default == literal))
        for guarantee in value.guarantees
    )


def redundant_guard[E](
    backend: Backend[E], predicate: Node, then: Lowered[E], default: Node, elements: Scope[E]
) -> bool:
    if not isinstance(predicate, Call):
        return False
    options = backend.rules.options(predicate.op.key)
    if not options.false_exact or not selects_default(then, predicate, default):
        return False
    inputs = [
        *predicate.arguments,
        *(
            value
            for expression in options.expressions.values()
            if (value := expression(predicate)) is not None
        ),
    ]
    return all(backend.exact(backend.lower(value, elements)) is None for value in inputs)


def selected[E](node: If, then: Lowered[E], orelse: Lowered[E]) -> tuple[Guarantee, ...]:
    found: list[Guarantee] = []
    for left, right, other in ((then, orelse, node.orelse), (orelse, then, node.then)):
        literal = constant(other)
        for guarantee in left.guarantees:
            alternatives = [
                other.default
                for other in right.guarantees
                if other.predicate == guarantee.predicate
            ]
            if literal is not None:
                alternatives.append(literal)
            for default in alternatives:
                if guarantee.default is None:
                    merged = Guarantee(guarantee.predicate, default)
                elif default is None or guarantee.default == default:
                    merged = guarantee
                else:
                    continue
                if merged not in found:
                    found.append(merged)
    return tuple(found)
