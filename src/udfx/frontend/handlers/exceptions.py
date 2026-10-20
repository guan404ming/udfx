from __future__ import annotations

import dis

from udfx.errors import Unsupported
from udfx.frontend.context import Context
from udfx.frontend.machine import handles
from udfx.frontend.state import State, Step
from udfx.frontend.values import Item, Opaque, Value, describe
from udfx.ir import lit


def exception_classes(item: Item) -> tuple[type[BaseException], ...]:
    candidates = item.value if isinstance(item, Opaque) else None
    members = candidates if isinstance(candidates, tuple) else (candidates,)
    classes = tuple(
        member
        for member in members
        if isinstance(member, type) and issubclass(member, BaseException)
    )
    if not classes or len(classes) != len(members):
        raise Unsupported(f"exception match against {describe(item)}")
    return classes


def exception_kind(item: Item) -> type[Exception]:
    match item:
        case Opaque(value=type() as kind) if issubclass(kind, Exception):
            return kind
        case _:
            raise Unsupported(f"expected an exception, got {describe(item)}")


@handles("PUSH_EXC_INFO")
def push_exc_info(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (exception,), rest = state.pop()
    return [rest.push(Opaque(None), exception).advance()]


@handles("POP_EXCEPT")
def pop_except(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.pop()[1].advance()]


@handles("CHECK_EXC_MATCH")
def check_exc_match(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (exception, match), rest = state.pop(2)
    matched = issubclass(exception_kind(exception), exception_classes(match))
    return [rest.push(exception, Value(lit(matched))).advance()]


@handles("RERAISE")
def reraise(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (exception,), rest = state.pop()
    return [context.raised(rest, exception_kind(exception))]
