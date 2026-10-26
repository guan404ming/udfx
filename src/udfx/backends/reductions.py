from __future__ import annotations

from typing import TYPE_CHECKING

from udfx.backends.hoisting import exact_free, hoist, invariant
from udfx.backends.indexing import UNROLL_LIMIT, literal_range
from udfx.backends.lowering_context import bindings_enabled
from udfx.backends.rules import Lowered, Scope
from udfx.frontend.core.logic import fold
from udfx.frontend.core.reductions import substitute
from udfx.ir import FALSE, Call, Comprehension, If, Lit, lit
from udfx.ops import apply
from udfx.types import INT

if TYPE_CHECKING:
    from udfx.backends.base import Backend


def lower_reduction[E](backend: Backend[E], node: Call, elements: Scope[E]) -> Lowered[E] | None:
    if node.op.name != "sum":
        return None
    source = node.arguments[0]
    if not isinstance(source, Comprehension) or source.fault != FALSE or source.value.type != INT:
        return None
    unrolled = source if isinstance(source.source, Lit) else literal_range(source)
    if unrolled is None or not isinstance(unrolled.source, Lit):
        return None
    items = unrolled.source.value
    if not isinstance(items, tuple | list) or len(items) > UNROLL_LIMIT:
        return None
    hoisted = invariant(source, elements) if bindings_enabled(backend.binds) else None
    if hoisted is not None and exact_free(backend, hoisted, elements):
        return hoist(backend, node, elements, hoisted)
    result = lit(0)
    for item in items:
        element = Lit(item, unrolled.element.type)
        condition = substitute(unrolled.condition, unrolled.element, element)
        value = substitute(unrolled.value, unrolled.element, element)
        result = apply("__add__", result, If(condition, value, lit(0), INT))
    return backend.lower(fold(result), elements)
