from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from udfx.backends.rules import Lowered
from udfx.ir import Call, Node
from udfx.semantics.arithmetic import IntegerRegionPrimitives, integer_region

if TYPE_CHECKING:
    from udfx.backends.base import Backend


def lower_integer_region[E](
    backend: Backend[E],
    node: Node,
    elements: Mapping[Node, E],
    primitives: IntegerRegionPrimitives,
) -> Lowered[E] | None:
    region = integer_region(node, primitives.maximum, primitives.size_limit)
    if not region:
        return None
    expressions: dict[Node, E] = {}
    exacts: list[E | None] = []

    def lower(part: Node) -> E:
        if part not in expressions:
            if part in region:
                call = cast("Call", part)
                left, right = (lower(argument) for argument in call.arguments)
                expressions[part] = primitives.operations[call.op.name](left, right)
            else:
                value = backend.lower(part, elements)
                exacts.append(backend.exact(value))
                expressions[part] = primitives.widen(value.expr)
        return expressions[part]

    wide = lower(node)
    result = primitives.narrow(wide)
    overflow = backend.both(backend.negate(primitives.not_null(result)), primitives.not_null(wide))
    return Lowered(result, backend.conjunction(*exacts), overflow)
