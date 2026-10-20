from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from functools import reduce
from types import FunctionType

from udfx.errors import Unsupported
from udfx.frontend.constants import constant
from udfx.frontend.context import Context, parameters
from udfx.frontend.handlers import explore
from udfx.frontend.logic import disjunction
from udfx.frontend.merge import merge
from udfx.frontend.state import Outcome, Raise, Return, State
from udfx.frontend.values import Item, Value, named
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
    if list(schema) != names:
        raise Unsupported(f"schema {list(schema)} does not match parameters {names}")
    context = Context.from_code(
        function.__code__, function.__globals__, function.__builtins__, None
    )
    columns = {name: Value(Col(name, type)) for name, type in schema.items()}
    start = State(0, (), {**columns, **_closure(function)}, TRUE)
    return _merge(function, schema, explore(context, start))


def _closure(function: FunctionType) -> dict[str, Item]:
    items: dict[str, Item] = {}
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True):
        try:
            items[name] = constant(cell.cell_contents)
        except ValueError:
            continue
    return items


def _merge(
    function: Callable[..., object], schema: Mapping[str, Type], outcomes: list[Outcome]
) -> Program:
    returns = [outcome for outcome in outcomes if isinstance(outcome, Return)]
    if not returns:
        raise Unsupported("function never returns")
    faults = [outcome.path for outcome in outcomes if isinstance(outcome, Raise)]
    return Program(
        function, dict(schema), merge(returns, faults), reduce(disjunction, faults, FALSE)
    )
