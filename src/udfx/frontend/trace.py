from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Literal

from udfx.errors import Unsupported
from udfx.frontend.core.bindings import Bindings
from udfx.frontend.core.branches import factor_branches
from udfx.frontend.core.evaluation import Raise, Return
from udfx.frontend.core.logic import fold
from udfx.frontend.core.merge import fault_union, merge
from udfx.frontend.core.series import require_reduced
from udfx.frontend.core.tracing import collect_effects, joining_helpers, tracing_series
from udfx.frontend.core.values import Item, Row, Series, Value, named
from udfx.frontend.flow.frame import Flow, Frame
from udfx.frontend.flow.machine import PathLimit, explore
from udfx.frontend.resolve.frozen import Snapshot, closure
from udfx.frontend.resolve.functions import parameters
from udfx.ir import TRUE, Col, If, Node
from udfx.ops import apply
from udfx.program import Program
from udfx.semantics.json_values import RESULTS, result_name
from udfx.types import Kind, Type, unify

SERIES_KINDS = (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL)
type Pandas = Literal["scalar", "aggregate"]


def trace(
    function: Callable[..., object],
    schema: Mapping[str, Type],
    *,
    result: Type | None = None,
    pandas: Pandas | None = None,
) -> Program:
    try:
        return trace_function(function, schema, result, pandas)
    except PathLimit:
        with joining_helpers():
            return trace_function(function, schema, result, pandas)


def trace_function(
    function: Callable[..., object],
    schema: Mapping[str, Type],
    result: Type | None = None,
    pandas: Pandas | None = None,
) -> Program:
    if not inspect.isfunction(function):
        raise Unsupported(f"{named(function)} is not a Python function")
    for name, column_type in schema.items():
        if not isinstance(column_type, Type):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise Unsupported(
                f"schema value for {name!r} must be a udfx.types.Type such as STR or INT.or_none()"
            )
    if function.__defaults__ or function.__kwdefaults__:
        raise Unsupported(f"{function.__qualname__} has parameter defaults")
    names = parameters(function.__code__)
    row_mode = pandas is None and len(names) == 1 and list(schema) != names
    if not row_mode and list(schema) != names:
        raise Unsupported(f"schema {list(schema)} does not match parameters {names}")
    snapshot = Snapshot()
    function = snapshot.function(function)
    frame = Frame.from_code(
        function.__code__, function.__globals__, function.__builtins__, None, snapshot
    )
    columns = {name: Value(Col(name, type)) for name, type in schema.items()}
    bindings = (
        series_columns(schema) if pandas else {names[0]: Row(columns)} if row_mode else columns
    )
    names = Bindings({**bindings, **closure(function)}, TRUE)
    with (
        collect_effects() as effects,
        tracing_series(pandas == "aggregate") if pandas else nullcontext() as reductions,
    ):
        outcomes = explore(frame, Flow(frame.decoded.entry, 0, names, {}))
    returns = [outcome for outcome in outcomes if isinstance(outcome, Return)]
    if not returns:
        raise Unsupported("function never returns")
    faults = [outcome.path for outcome in outcomes if isinstance(outcome, Raise)]
    value = merge(returns, faults)
    if value.type.kind is Kind.JSON:
        if result is None or result.strict() not in RESULTS:
            raise Unsupported("dynamic JSON returns require a scalar or scalar-list result type")
        value = fold(json_result(value, result))
    elif result is not None and unify(value.type, result) != result:
        raise Unsupported(f"inferred result {value.type} does not fit declared result {result}")
    if value.type.item is not None and value.type.item.kind is Kind.JSON:
        raise Unsupported("dynamic JSON lists require a scalar result conversion")
    fault = fault_union(faults)
    if pandas == "aggregate":
        require_reduced(value, fault)
    if not effects:
        value, fault = factor_branches((value, fault))
    return Program(
        snapshot.callable(function),
        dict(schema),
        value,
        fault,
        row_mode,
        not effects,
        sys.modules["pandas"].Series if pandas else None,
        tuple(reductions) if pandas == "aggregate" and reductions is not None else None,
    )


def series_columns(schema: Mapping[str, Type]) -> dict[str, Item]:
    if "pandas" not in sys.modules:
        raise Unsupported("series UDFs require pandas")
    for name, column_type in schema.items():
        if column_type.nullable or column_type.kind not in SERIES_KINDS:
            raise Unsupported(f"series column {name!r} must be a non-null str, int, float, or bool")
    return {name: Series(Col(name, type)) for name, type in schema.items()}


def json_result(value: Node, result: Type) -> Node:
    seen: dict[int, Node] = {}

    def visit(node: Node) -> Node:
        if id(node) not in seen:
            seen[id(node)] = (
                If(node.condition, visit(node.then), visit(node.orelse), result)
                if isinstance(node, If)
                else apply(result_name(result), node)
            )
        return seen[id(node)]

    return visit(value)
