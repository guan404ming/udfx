from __future__ import annotations

import dis
from collections.abc import Sequence
from types import CodeType, ModuleType

from udfx.errors import Unsupported
from udfx.frontend.calls import call
from udfx.frontend.constants import constant
from udfx.frontend.context import Context, parameters
from udfx.frontend.machine import emit, explore, handles
from udfx.frontend.state import Return, State, Step
from udfx.frontend.values import (
    NULL,
    Cell,
    Function,
    Item,
    Method,
    Null,
    Opaque,
    Row,
    Value,
    describe,
)

CLOSURE = 0x08
ANNOTATIONS = 0x04


@handles("PUSH_NULL")
def push_null(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.push(NULL).advance()]


@handles("LOAD_GLOBAL")
def load_global(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    item = context.global_(instruction.argval)
    pushed = (
        state.push(NULL, item)
        if instruction.arg is not None and instruction.arg & 1
        else state.push(item)
    )
    return [pushed.advance()]


@handles("LOAD_ATTR")
def load_attr(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    name = instruction.argval
    method = instruction.arg is not None and instruction.arg & 1
    (owner,), rest = state.pop()
    match owner:
        case Row():
            attribute = owner.column(name)
        case Opaque(value=ModuleType() as module) if name in vars(module):
            attribute = constant(vars(module)[name])
        case Value() if method:
            return [rest.push(Method(name), owner).advance()]
        case Value():
            raise Unsupported(
                f"attribute access on a column value (attribute load {name})",
                context.where(state),
            )
        case _:
            raise Unsupported(f"attribute load {name} of {describe(owner)}", context.where(state))
    return [(rest.push(NULL, attribute) if method else rest.push(attribute)).advance()]


@handles("KW_NAMES")
def kw_names(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    raise Unsupported("keyword arguments", context.where(state))


@handles("CALL")
def call_(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    count = instruction.arg or 0
    items, rest = state.pop(count + 2)
    head, second, *others = items
    callee, arguments = (second, others) if isinstance(head, Null) else (head, [second, *others])
    if any(isinstance(argument, Row) for argument in arguments):
        raise Unsupported("passing a row to a call", context.where(state))
    if isinstance(callee, Function):
        return inline(context, rest, callee, arguments)
    node, faults = call(callee, arguments)
    return emit(context, rest, Value(node), faults)


def inline(
    context: Context, state: State, function: Function, arguments: Sequence[Item]
) -> list[Step]:
    code = function.code
    frame = context.frame(state, function.home)
    if context.encloses(code):
        raise Unsupported(f"{code.co_qualname} is recursive")
    names = parameters(code)
    if len(arguments) != len(names):
        raise Unsupported(f"{code.co_qualname} takes {len(names)} arguments")
    free = {
        name: frame.load(cell) for name, cell in zip(code.co_freevars, function.cells, strict=True)
    }
    start = State(0, (), {**dict(zip(names, arguments, strict=True)), **free}, state.path)
    steps: list[Step] = []
    for outcome in explore(context.nested(code, state), start):
        under = state.under(outcome.path)
        if isinstance(outcome, Return):
            steps.append(under.push(Value(outcome.value)).advance())
        else:
            steps.append(context.raised(under, outcome.kind))
    return steps


@handles("MAKE_FUNCTION")
def make_function(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    flags = instruction.arg or 0
    if flags & ~(CLOSURE | ANNOTATIONS):
        raise Unsupported("helper with parameter defaults", context.where(state))
    (code,), rest = state.pop()
    cells: tuple[str, ...] = ()
    if flags & CLOSURE:
        (closure,), rest = rest.pop()
        cells = cell_names(closure)
    if flags & ANNOTATIONS:
        rest = rest.pop()[1]
    if not isinstance(code, Opaque) or not isinstance(code.value, CodeType):
        raise Unsupported(f"function from {describe(code)}", context.where(state))
    return [rest.push(Function(code.value, cells, context)).advance()]


def cell_names(item: Item) -> tuple[str, ...]:
    members = item.value if isinstance(item, Opaque) else None
    if not isinstance(members, tuple):
        raise Unsupported(f"closure {describe(item)}")
    names = tuple(member.name for member in members if isinstance(member, Cell))
    if len(names) != len(members):
        raise Unsupported(f"closure {describe(item)}")
    return names


@handles("LOAD_CLOSURE")
def load_closure(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.push(Opaque(Cell(instruction.argval))).advance()]


@handles("LOAD_DEREF")
def load_deref(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.push(state.load(instruction.argval)).advance()]


@handles("STORE_DEREF")
def store_deref(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    if instruction.argval in context.code.co_freevars:
        raise Unsupported(f"nonlocal write to {instruction.argval}", context.where(state))
    (item,), rest = state.pop()
    return [rest.store(instruction.argval, item).advance()]


@handles("MAKE_CELL", "COPY_FREE_VARS")
def prepare_cells(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    return [state.advance()]
