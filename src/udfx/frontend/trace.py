from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from functools import reduce

from udfx.errors import Unsupported
from udfx.frontend.context import Context, parameters
from udfx.frontend.frozen import Snapshot, closure
from udfx.frontend.handlers import explore
from udfx.frontend.logic import disjunction
from udfx.frontend.merge import merge
from udfx.frontend.state import Outcome, Raise, Return, State
from udfx.frontend.values import Row, Value, named
from udfx.ir import FALSE, TRUE, Col
from udfx.program import Program
from udfx.types import Type


def trace(function: Callable[..., object], schema: Mapping[str, Type]) -> Program:
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
    row_mode = len(names) == 1 and list(schema) != names
    if not row_mode and list(schema) != names:
        raise Unsupported(f"schema {list(schema)} does not match parameters {names}")
    snapshot = Snapshot()
    function = snapshot.function(function)
    context = Context.from_code(
        function.__code__, function.__globals__, function.__builtins__, None, snapshot
    )
    columns = {name: Value(Col(name, type)) for name, type in schema.items()}
    bindings = {names[0]: Row(columns)} if row_mode else columns
    start = State(0, (), {**bindings, **closure(function)}, TRUE)
    return _merge(function, schema, explore(context, start), row_mode)


def _merge(
    function: Callable[..., object],
    schema: Mapping[str, Type],
    outcomes: list[Outcome],
    row_mode: bool,
) -> Program:
    returns = [outcome for outcome in outcomes if isinstance(outcome, Return)]
    if not returns:
        raise Unsupported("function never returns")
    faults = [outcome.path for outcome in outcomes if isinstance(outcome, Raise)]
    return Program(
        function, dict(schema), merge(returns, faults), reduce(disjunction, faults, FALSE), row_mode
    )
