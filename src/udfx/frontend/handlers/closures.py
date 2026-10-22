from __future__ import annotations

import dis
from collections.abc import Mapping, Sequence
from types import CodeType, FunctionType
from typing import cast

from udfx.errors import Unsupported
from udfx.frontend.binding import bind_helper, helper_signature
from udfx.frontend.constants import constant, constant_value, scalar_map
from udfx.frontend.context import Context
from udfx.frontend.frozen import closure
from udfx.frontend.handlers.iteration import comprehension
from udfx.frontend.iterables import source_of
from udfx.frontend.machine import explore, handles
from udfx.frontend.state import Return, State, Step
from udfx.frontend.values import Cell, Function, Generator, Item, Iterator, Opaque, Value, describe
from udfx.ir import Lit

DEFAULTS = 0x01
KEYWORD_DEFAULTS = 0x02
CLOSURE = 0x08
ANNOTATIONS = 0x04
MAX_DEPTH = 32


def inline(
    context: Context,
    state: State,
    function: Function,
    arguments: Sequence[Item],
    keywords: Mapping[str, Item],
) -> list[Step]:
    code = function.code
    if context.depth >= MAX_DEPTH:
        raise Unsupported(f"recursive helper {code.co_qualname}")
    defaults = function.defaults
    keyword_defaults = function.keyword_defaults
    if isinstance(function.home, FunctionType):
        helper = function.home
        defaults = tuple(constant(value) for value in helper.__defaults__ or ())
        keyword_defaults = {
            name: constant(value) for name, value in (helper.__kwdefaults__ or {}).items()
        }
        free = closure(helper)
        nested = Context.from_code(
            code, helper.__globals__, helper.__builtins__, (context, state), context.snapshot
        )
    else:
        frame = context.frame(state, function.home)
        home = cast("Context", function.home)
        free = {
            name: frame.load(cell)
            for name, cell in zip(code.co_freevars, function.cells, strict=True)
        }
        nested = Context.from_code(
            code, home.scope, home.builtins, (context, state), context.snapshot
        )
    if code.co_name == "<genexpr>" and len(arguments) == 1 and isinstance(arguments[0], Iterator):
        iterator = arguments[0]
        index = next(
            index
            for index, instruction in enumerate(nested.instructions)
            if instruction.opname == "FOR_ITER"
        )
        start = State(index, (iterator,), {".0": iterator, **free}, state.path)
        node = (
            comprehension(nested, start, iterator, "YIELD_VALUE")
            if iterator.items or iterator.source is not None
            else source_of(iterator)
        )
        return [state.push(Generator(node)).advance()]
    for item in (*defaults, *keyword_defaults.values()):
        if isinstance(item, Opaque) and isinstance(item.value, dict):
            scalar_map(item.value.items())
        elif not isinstance(item, Value) or not isinstance(item.node, Lit):
            raise Unsupported("helper with non-constant parameter defaults")
    signature = helper_signature(code, defaults, keyword_defaults)
    bindings = bind_helper(signature, arguments, keywords)
    start = State(0, (), {**bindings, **free}, state.path)
    steps: list[Step] = []
    for outcome in explore(nested, start):
        under = state.under(outcome.path)
        if isinstance(outcome, Return):
            steps.append(under.push(Value(outcome.value)).advance())
        else:
            steps.append(context.raised(under, outcome.kind))
    return steps


@handles("MAKE_FUNCTION")
def make_function(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    flags = instruction.arg or 0
    if flags & ~(CLOSURE | ANNOTATIONS | DEFAULTS | KEYWORD_DEFAULTS):
        raise Unsupported("helper with parameter defaults", context.where(state))
    (code,), rest = state.pop()
    cells: tuple[str, ...] = ()
    if flags & CLOSURE:
        (closure,), rest = rest.pop()
        cells = cell_names(closure)
    if flags & ANNOTATIONS:
        rest = rest.pop()[1]
    keyword_defaults: dict[str, Item] = {}
    if flags & KEYWORD_DEFAULTS:
        (item,), rest = rest.pop()
        values = constant_value(item)
        if not isinstance(values, dict):
            raise Unsupported("helper with non-constant parameter defaults")
        keyword_defaults = {name: constant(value) for name, value in values.items()}
    defaults: tuple[Item, ...] = ()
    if flags & DEFAULTS:
        (item,), rest = rest.pop()
        values = constant_value(item)
        if not isinstance(values, tuple):
            raise Unsupported("helper with non-constant parameter defaults")
        defaults = tuple(constant(value) for value in values)
    if not isinstance(code, Opaque) or not isinstance(code.value, CodeType):
        raise Unsupported(f"function from {describe(code)}", context.where(state))
    return [rest.push(Function(code.value, cells, context, defaults, keyword_defaults)).advance()]


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
