from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from udfx.backends.rules import Lowered, Scope
from udfx.ir import Call, Lit, Node
from udfx.semantics.numbers.regions import IntegerRegionPrimitives, integer_region

if TYPE_CHECKING:
    from udfx.backends.base import Backend


def lower_integer_region[E](
    backend: Backend[E],
    node: Node,
    elements: Scope[E],
    primitives: IntegerRegionPrimitives,
) -> Lowered[E] | None:
    region = integer_region(node, primitives.maximum, primitives.size_limit)
    if region is None:
        return None
    expressions: dict[Node, E] = {}
    exacts: list[E | Literal[False] | None] = []
    present: list[E] = []

    def lower(part: Node) -> E:
        if part not in expressions:
            if part in region.operations:
                call = cast("Call", part)
                left, right = (lower(argument) for argument in call.arguments)
                expressions[part] = primitives.operations[call.op.name](left, right)
            else:
                value = backend.lower(part, elements)
                exacts.append(backend.exact(value))
                if not isinstance(part, Lit):
                    present.append(primitives.not_null(value.expr))
                expressions[part] = value.expr if region.narrow else primitives.widen(value.expr)
        return expressions[part]

    computed = lower(node)
    if region.narrow:
        result, defined = computed, backend.predicate(backend.conjunction(*present))
    else:
        result, defined = primitives.narrow(computed), primitives.not_null(computed)
    overflow = backend.both(backend.negate(primitives.not_null(result)), defined)
    return Lowered(result, backend.conjunction(*exacts), overflow)
