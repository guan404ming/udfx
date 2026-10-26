from __future__ import annotations

from typing import TYPE_CHECKING

from udfx.backends.rules import Lowered, Scope, fold_predicate
from udfx.frontend.core.logic import disjunction
from udfx.ir import If, Lit, Node

if TYPE_CHECKING:
    from udfx.backends.base import Backend


def lower_cases[E](backend: Backend[E], node: If, elements: Scope[E]) -> Lowered[E] | None:
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
