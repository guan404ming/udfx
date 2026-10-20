from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from functools import reduce

from udfx.errors import Unsupported
from udfx.frontend.context import Context, parameters
from udfx.frontend.handlers import explore
from udfx.frontend.logic import choose, disj
from udfx.frontend.state import Outcome, Raise, Return, State
from udfx.frontend.values import Value
from udfx.ir import FALSE, TRUE, Col
from udfx.program import Program
from udfx.types import Type


def trace(function: Callable[..., object], schema: Mapping[str, Type]) -> Program:
    if not inspect.isfunction(function):
        raise Unsupported(f"{function!r} is not a Python function")
    if function.__defaults__ or function.__kwdefaults__:
        raise Unsupported(f"{function.__qualname__} has parameter defaults")
    names = parameters(function.__code__)
    if list(schema) != names:
        raise Unsupported(f"schema {list(schema)} does not match parameters {names}")
    context = Context.of(function.__code__, function.__globals__, function.__builtins__)
    start = State(0, (), {name: Value(Col(name, type)) for name, type in schema.items()}, TRUE)
    return _merge(function, schema, explore(context, start))


def _merge(
    function: Callable[..., object], schema: Mapping[str, Type], outcomes: list[Outcome]
) -> Program:
    returns = [(outcome.path, outcome.value) for outcome in outcomes if isinstance(outcome, Return)]
    if not returns:
        raise Unsupported("function never returns")
    fault = reduce(disj, (o.path for o in outcomes if isinstance(o, Raise)), FALSE)
    return Program(function, dict(schema), choose(returns), fault)
