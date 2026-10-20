from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from functools import reduce

from udfx.errors import Unsupported
from udfx.frontend.context import Context
from udfx.frontend.handlers import HANDLERS
from udfx.frontend.logic import disj
from udfx.frontend.state import Outcome, Raise, Return, State
from udfx.frontend.values import Value
from udfx.ir import FALSE, TRUE, Col, If, Lit, Node
from udfx.program import Program
from udfx.types import Kind, Type, unify

MAX_STEPS = 10_000


def trace(function: Callable[..., object], schema: Mapping[str, Type]) -> Program:
    parameters = _parameters(function)
    if list(schema) != parameters:
        raise Unsupported(f"schema {list(schema)} does not match parameters {parameters}")
    ctx = Context.of(function)
    start = State(0, (), {name: Value(Col(name, type)) for name, type in schema.items()}, TRUE)
    return _merge(function, schema, _explore(ctx, start))


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


def _explore(ctx: Context, start: State) -> list[Outcome]:
    pending: list[State] = [start]
    outcomes: list[Outcome] = []
    steps = 0
    while pending:
        steps += 1
        if steps > MAX_STEPS:
            raise Unsupported("path explosion")
        state = pending.pop()
        inst = ctx.instructions[state.index]
        handler = HANDLERS.get(inst.opname)
        if handler is None:
            raise Unsupported(f"opcode {inst.opname}", ctx.where(state))
        for step in reversed(handler(ctx, state, inst)):
            if isinstance(step, State):
                pending.append(step)
            else:
                outcomes.append(step)
    return outcomes


def _merge(
    function: Callable[..., object], schema: Mapping[str, Type], outcomes: list[Outcome]
) -> Program:
    returns = [outcome for outcome in outcomes if isinstance(outcome, Return)]
    if not returns:
        raise Unsupported("function never returns")
    result = reduce(_unify, (outcome.value.type for outcome in returns))
    value = _retype(returns[-1].value, result)
    for outcome in reversed(returns[:-1]):
        value = If(outcome.path, _retype(outcome.value, result), value, result)
    fault = reduce(disj, (o.path for o in outcomes if isinstance(o, Raise)), FALSE)
    return Program(function, dict(schema), value, fault)


def _unify(a: Type, b: Type) -> Type:
    unified = unify(a, b)
    if unified is None:
        raise Unsupported(f"return types {a} and {b} differ")
    return unified


def _retype(node: Node, type: Type) -> Node:
    if isinstance(node, Lit) and node.type.kind is Kind.NONE:
        return Lit(None, type.or_none())
    return node
