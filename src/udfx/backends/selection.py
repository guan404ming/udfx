from __future__ import annotations

from typing import TYPE_CHECKING

from udfx.backends.rules import Lowered, Scope, fold_predicate
from udfx.frontend.core.logic import disjunction
from udfx.ir import Call, If, Lit, Node
from udfx.types import Kind

if TYPE_CHECKING:
    from udfx.backends.base import Backend


def lower_cases[E](backend: Backend[E], node: If, elements: Scope[E]) -> Lowered[E] | None:
    selected = lower_literal_cases(backend, node, elements)
    if selected is not None:
        return selected
    cases = []
    current: Node = node
    while isinstance(current, If):
        if not isinstance(current.then, Lit):
            return None
        condition = fold_predicate(current.condition, backend.rules.rewrite_call)
        if (
            cases
            and isinstance(current.then.value, str | int | bool | type(None))
            and cases[-1][1] == current.then
        ):
            previous, value = cases.pop()
            cases.append((disjunction(previous, condition), value))
        else:
            cases.append((condition, current.then))
        current = current.orelse
    if len(cases) < 2 or not isinstance(current, Lit):
        return None
    lowered = [
        (backend.lower(condition, elements), backend.lower(value, elements))
        for condition, value in cases
    ]
    default = backend.lower(current, elements)
    parts = [default, *(part for case in lowered for part in case)]
    if any(part.exact is not None or part.overflow is not None for part in parts):
        return None
    return Lowered(
        backend.select_many(
            [(condition.expr, value.expr) for condition, value in lowered], default.expr
        )
    )


def lower_literal_cases[E](backend: Backend[E], node: If, elements: Scope[E]) -> Lowered[E] | None:
    source = None
    cases: list[tuple[Lit, Lit]] = []
    current: Node = node
    while isinstance(current, If):
        condition = current.condition
        if (
            not isinstance(condition, Call)
            or condition.op.name != "__eq__"
            or not isinstance(current.then, Lit)
            or current.then.value is None
            or current.then.type.kind != node.type.kind
        ):
            return None
        value, key = condition.arguments
        if (
            not isinstance(key, Lit)
            or key.type.kind not in {Kind.STR, Kind.INT}
            or key.value is None
            or value.type.kind != key.type.kind
            or (source is not None and value != source)
            or any(previous == key for previous, _ in cases)
        ):
            return None
        source = value
        cases.append((key, current.then))
        current = current.orelse
    if source is None or len(cases) < 2 or not isinstance(current, Lit):
        return None
    lowered = backend.lower(source, elements)
    expr = backend.select_value(
        lowered.expr,
        [(backend.literal(key), backend.literal(value)) for key, value in cases],
        backend.literal(current),
    )
    return None if expr is None else Lowered(expr, backend.exact(lowered))
