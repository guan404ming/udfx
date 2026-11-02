from __future__ import annotations

from collections.abc import Sequence

from udfx.errors import Unsupported
from udfx.frontend.core.expressions import invoke, node_of
from udfx.frontend.core.values import Item, Iterator, Opaque, Value, require_bounded
from udfx.frontend.resolve.constants import constant_value
from udfx.ir import Lit, lit
from udfx.ops import Fault


def constant_range(arguments: Sequence[Item]) -> Opaque:
    values = [constant_value(argument) for argument in arguments]
    if not 1 <= len(values) <= 3 or not all(isinstance(value, int) for value in values):
        raise Unsupported("range requires one to three integer literals")
    sequence = range(*(value for value in values if isinstance(value, int)))
    if sequence:
        require_bounded((sequence[-1] - sequence[0]) // sequence.step + 1)
    return Opaque(sequence)


def range_iterator(arguments: Sequence[Item]) -> tuple[Item, list[Fault]]:
    if all(
        isinstance(argument, Value) and isinstance(argument.node, Lit) for argument in arguments
    ):
        return constant_range(arguments), []
    if not 1 <= len(arguments) <= 3:
        raise Unsupported("range requires one to three arguments")
    nodes = [node_of(argument) for argument in arguments]
    if len(nodes) > 1:
        if not isinstance(nodes[0], Lit) or (len(nodes) == 3 and not isinstance(nodes[2], Lit)):
            raise Unsupported("range start and step require integer literals")
        if len(nodes) == 2:
            nodes.append(lit(1))
    source, faults = invoke("range", nodes, TypeError)
    return Iterator((), source=source, requires_native=True), faults
