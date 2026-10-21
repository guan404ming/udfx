from __future__ import annotations

import dis
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import CodeType, FunctionType, ModuleType
from typing import cast

from udfx.errors import Unsupported
from udfx.frontend.binding import bind, helper_signature
from udfx.frontend.calls import call, invoke, iterable_node, node_of
from udfx.frontend.constants import (
    constant,
    constant_value,
    lookup,
    runtime_type,
    scalar_map,
    type_test,
)
from udfx.frontend.context import Context
from udfx.frontend.frozen import closure
from udfx.frontend.handlers.iteration import comprehension, source_of
from udfx.frontend.machine import emit, explore, handles
from udfx.frontend.regex import call_regex
from udfx.frontend.state import Return, State, Step
from udfx.frontend.values import (
    NULL,
    Cell,
    Function,
    Generator,
    Item,
    Iterator,
    Match,
    Method,
    Null,
    Opaque,
    Row,
    Set,
    Value,
    describe,
)
from udfx.ir import Lit, lit
from udfx.types import STR

DEFAULTS = 0x01
KEYWORD_DEFAULTS = 0x02
CLOSURE = 0x08
ANNOTATIONS = 0x04
MAX_DEPTH = 32


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
        case Opaque(value=ModuleType() as module) if name in context.snapshot.members(module):
            attribute = constant(context.snapshot.attribute(module, name))
        case Opaque(value=dict()) if name == "get" and method:
            return [rest.push(Method(name), owner).advance()]
        case Opaque(value=re.Pattern() as pattern):
            attribute = Opaque(getattr(pattern, name, None))
        case Value() | Match() if method:
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
    names = instruction.argval
    if not isinstance(names, tuple) or not all(isinstance(name, str) for name in names):
        raise Unsupported("keyword argument names", context.where(state))
    if len(set(names)) != len(names):
        raise Unsupported("duplicate keyword arguments", context.where(state))
    return [replace(state, keywords=names).advance()]


@handles("CALL")
def call_(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    count = instruction.arg or 0
    items, rest = state.pop(count + 2)
    head, second, *others = items
    callee, arguments = (second, others) if isinstance(head, Null) else (head, [second, *others])
    if any(isinstance(argument, Row) for argument in arguments):
        raise Unsupported("passing a row to a call", context.where(state))
    keywords: dict[str, Item] = {}
    if state.keywords:
        split = len(arguments) - len(state.keywords)
        keywords = dict(zip(state.keywords, arguments[split:], strict=True))
        arguments = arguments[:split]
        rest = replace(rest, keywords=())
    if isinstance(callee, Function):
        return inline(context, rest, callee, arguments, keywords)
    regex = call_regex(callee, arguments, keywords, state.path)
    if regex is not None:
        item, faults = regex
        return emit(context, rest, item, faults)
    if isinstance(callee, Opaque) and not keywords:
        if callee.value is set and len(arguments) == 1:
            source = iterable_node(arguments[0])
            if source.type.item != STR:
                raise Unsupported("set requires list[str]", context.where(state))
            node, faults = invoke("list", [source], TypeError)
            return emit(context, rest, Set(node), faults)
        if callee.value is type and len(arguments) == 1:
            return [rest.push(runtime_type(arguments[0])).advance()]
        if callee.value is isinstance and len(arguments) == 2:
            test = type_test(runtime_type(arguments[0]), constant_value(arguments[1]), True)
            return [rest.push(Value(test)).advance()]
    if isinstance(callee, Method) and callee.name == "get" and arguments and not keywords:
        owner, *parameters = arguments
        if isinstance(owner, Opaque) and isinstance(owner.value, dict):
            if len(parameters) not in (1, 2):
                raise Unsupported("dict.get takes one or two arguments", context.where(state))
            key, *defaults = map(node_of, parameters)
            node, faults = lookup(owner.value, key, defaults[0] if defaults else lit(None))
            return emit(context, rest, Value(node), faults)
    node, faults = call(callee, arguments, keywords)
    return emit(context, rest, Value(node), faults)


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
    bindings = bind(signature, arguments, keywords)
    bindings = {
        name: bindings.get(name, cast("Item", parameter.default))
        for name, parameter in signature.parameters.items()
    }
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
