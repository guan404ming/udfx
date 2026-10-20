from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from functools import reduce

from udfx.errors import Unsupported
from udfx.frontend.context import Context
from udfx.frontend.handlers import explore
from udfx.frontend.logic import choose, disj
from udfx.frontend.state import Outcome, Raise, Return, State
from udfx.frontend.values import Value
from udfx.ir import FALSE, TRUE, Col
from udfx.program import Program
from udfx.types import Type


def trace(function: Callable[..., object], schema: Mapping[str, Type]) -> Program:
    parameters = _parameters(function)
    if list(schema) != parameters:
        raise Unsupported(f"schema {list(schema)} does not match parameters {parameters}")
    ctx = Context.of(function)
    start = State(0, (), {name: Value(Col(name, type)) for name, type in schema.items()}, TRUE)
    return _merge(function, schema, explore(ctx, start))


def _parameters(function: Callable[..., object]) -> list[str]:
    if not inspect.isfunction(function):
        raise Unsupported(f"{function!r} is not a Python function")
    names: list[str] = []
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise Unsupported(f"parameter {parameter.name} is {parameter.kind.description}")
        if parameter.default is not inspect.Parameter.empty:
            raise Unsupported(f"parameter {parameter.name} has a default")
        names.append(parameter.name)
    return names


def _merge(
    function: Callable[..., object], schema: Mapping[str, Type], outcomes: list[Outcome]
) -> Program:
    returns = [(outcome.path, outcome.value) for outcome in outcomes if isinstance(outcome, Return)]
    if not returns:
        raise Unsupported("function never returns")
    fault = reduce(disj, (o.path for o in outcomes if isinstance(o, Raise)), FALSE)
    return Program(function, dict(schema), choose(returns), fault)
