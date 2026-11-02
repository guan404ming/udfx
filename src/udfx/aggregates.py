from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, cast

from udfx.errors import Unsupported
from udfx.frontend.core.logic import fold
from udfx.frontend.core.series import AGGREGATE
from udfx.graphs import map_children
from udfx.ir import FALSE, TRUE, Col, Lit, Node
from udfx.ops import OpKey
from udfx.program import Program, Reduction
from udfx.semantics.scalars import numpy_values

if TYPE_CHECKING:
    from udfx.backends.base import Backend
    from udfx.backends.rules import Lowered

SKIPPING = (
    frozenset({"sum", "mean", "var", "std", "max", "min", "corr"})
    if find_spec("bottleneck") is None
    else frozenset({"corr"})
)


@dataclass(frozen=True)
class Aggregate[E]:
    program: Program
    columns: tuple[E, ...]
    inputs: tuple[E, ...]
    reduced: bool = False

    def finish(self, columns: Sequence[Any], inputs: Callable[[], Sequence[object]]) -> object:
        program = self.program
        series = program.series
        assert series is not None
        if self.reduced:
            values = [plain(column[0]) for column in columns]
            settled = not any(unsettled(value) for value in values)
        else:
            found = dict(zip(needed(program), columns, strict=True))
            values = [reduce_series(program, part, found) for part in program.reductions or ()]
            settled = True
        reduced = reduce_columns(program, values) if settled else None
        if reduced is not None:
            return reduced
        return plain(program.function(*(series(column) for column in inputs())))


def unsettled(value: object) -> bool:
    return value is None or (isinstance(value, float) and (math.isnan(value) or value == 0.0))


def needed(program: Program) -> tuple[Node, ...]:
    nodes = (
        node
        for reduction in program.reductions or ()
        for node in (reduction.present, *(() if reduction.method == "len" else reduction.values))
    )
    return tuple(dict.fromkeys(node for node in nodes if node != TRUE)) or (TRUE,)


def reduce_columns(program: Program, values: Sequence[object]) -> object | None:
    literals = {
        f"{AGGREGATE}{index}": Lit(value, reduction.type)
        for index, (value, reduction) in enumerate(
            zip(values, program.reductions or (), strict=True)
        )
    }
    seen: dict[int, Node] = {}

    def visit(node: Node) -> Node:
        if id(node) not in seen:
            seen[id(node)] = (
                literals[node.name]
                if isinstance(node, Col) and node.name in literals
                else map_children(node, visit)
            )
        return seen[id(node)]

    value, fault = fold(visit(program.value)), fold(visit(program.fault))
    return value.value if fault == FALSE and isinstance(value, Lit) else None


def reduce_series(program: Program, reduction: Reduction, columns: dict[Node, Any]) -> object:
    series = program.series
    assert series is not None
    present = reduction.present
    mask = None if present == TRUE else series(columns[present]).to_numpy(dtype=bool)
    if reduction.method == "len":
        return len(next(iter(columns.values()))) if mask is None else int(mask.sum())
    parts = [
        series(columns[node] if mask is None else series(columns[node]).to_numpy()[mask])
        for node in reduction.values
    ]
    method = reduction.method
    if method in SKIPPING:
        found = plain(unchecked(parts, method))
        if not (isinstance(found, float) and math.isnan(found)):
            return found
    return plain(parts[0].corr(parts[1]) if method == "corr" else getattr(parts[0], method)())


def unchecked(parts: Sequence[Any], method: str) -> object:
    if method == "corr":
        return numpy_values("corrcoef", *(part.to_numpy(dtype=float) for part in parts))[0, 1]
    return getattr(parts[0], method)(skipna=False)


def build_aggregate[E](backend: Backend[E], program: Program) -> Aggregate[E]:
    def native(node: Node) -> E:
        def exact(part: Lowered[E]) -> E:
            if part.exact is not None:
                raise Unsupported("aggregate inputs require exact native expressions")
            return part.expr

        lowered = backend.lower_program(replace(program, value=node, fault=FALSE, reductions=None))
        return lowered.consume(exact)

    inputs = tuple(backend.column(Col(name, type)) for name, type in program.schema.items())
    calls = [
        (backend.rules.get(OpKey(f"aggregate {reduction.method}", kinds)), nodes)
        for reduction in program.reductions or ()
        for nodes in (
            (*(() if reduction.method == "len" else reduction.values), reduction.present),
        )
        for kinds in (tuple(node.type.kind for node in nodes),)
    ]
    if calls and all(rule is not None for rule, _ in calls):
        columns = tuple(
            cast("E", rule(*map(native, nodes))) for rule, nodes in calls if rule is not None
        )
        return Aggregate(program, columns, inputs, reduced=True)
    return Aggregate(program, tuple(native(node) for node in needed(program)), inputs)


def plain(value: object) -> object:
    item = getattr(value, "item", None)
    return item() if callable(item) else value
