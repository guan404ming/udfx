from __future__ import annotations

import dis
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from functools import reduce
from types import ModuleType

from udfx.errors import Unsupported
from udfx.frontend.calls import call, invoke, node_of, truth
from udfx.frontend.constants import (
    constant,
    constant_value,
    make_translation_table,
    runtime_type,
    type_test,
)
from udfx.frontend.context import Context
from udfx.frontend.datetimes import call_datetime, duration_attribute
from udfx.frontend.handlers.closures import inline
from udfx.frontend.handlers.iteration import validate_fault_handlers
from udfx.frontend.iterables import call_iterable, indexed, iterator_of, source_of
from udfx.frontend.logic import choose, conjunction, disjunction
from udfx.frontend.machine import emit, handles
from udfx.frontend.paths import call_path
from udfx.frontend.regex import call_regex
from udfx.frontend.state import Raise, State, Step
from udfx.frontend.values import (
    NULL,
    Duration,
    Function,
    Generator,
    Item,
    Match,
    Method,
    Null,
    Opaque,
    Row,
    Tuple,
    Value,
    describe,
)
from udfx.ir import FALSE, TRUE, Comprehension, Node, lit
from udfx.semantics.datetimes import ATTRIBUTES
from udfx.types import Kind


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
        case Opaque(value=python_type) if python_type is datetime and name == "strptime":
            attribute = Opaque(datetime.strptime)
        case Opaque(value=python_type) if python_type is str and name == "maketrans":
            attribute = Opaque(str.maketrans)
        case Opaque(value=dict()) if name in {"get", "keys", "values", "items"} and method:
            return [rest.push(Method(name), owner).advance()]
        case Opaque(value=re.Pattern() as pattern):
            attribute = Opaque(getattr(pattern, name, None))
        case Value(node=node) if node.type.kind is Kind.DATETIME and name in ATTRIBUTES:
            if method:
                raise Unsupported(
                    f"datetime attribute {name} is not callable", context.where(state)
                )
            result, faults = invoke(f".{name}", [node], AttributeError)
            return emit(context, rest, Value(result), faults)
        case Duration(node=node) if name in {"days", "seconds"} and not method:
            result, faults = duration_attribute(node, name)
            return emit(context, rest, Value(result), faults)
        case Value() | Match() | Duration() if method:
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
    keywords: dict[str, Item] = {}
    if state.keywords:
        split = len(arguments) - len(state.keywords)
        keywords = dict(zip(state.keywords, arguments[split:], strict=True))
        arguments = arguments[:split]
        rest = replace(rest, keywords=())
    return dispatch_call(context, rest, callee, arguments, keywords)


@handles("CALL_FUNCTION_EX")
def call_unpacked(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    rest = state
    if instruction.arg:
        (keywords,), rest = rest.pop()
        if constant_value(keywords) != {}:
            raise Unsupported("nonempty unpacked keyword arguments", context.where(state))
    (head, callee, arguments), rest = rest.pop(3)
    if not isinstance(head, Null):
        raise Unsupported("unpacked call arguments", context.where(state))
    if isinstance(arguments, Tuple):
        items = arguments.items
    else:
        values = constant_value(arguments)
        if not isinstance(values, tuple):
            raise Unsupported("unpacked call arguments require a tuple", context.where(state))
        items = tuple(constant(value) for value in values)
    return dispatch_call(context, rest, callee, items, {})


@handles("DICT_MERGE")
def merge_keywords(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    (keywords,), rest = state.pop()
    if constant_value(keywords) != {}:
        raise Unsupported("nonempty unpacked keyword arguments", context.where(state))
    return [rest.advance()]


def dispatch_call(
    context: Context,
    rest: State,
    callee: Item,
    arguments: Sequence[Item],
    keywords: Mapping[str, Item],
) -> list[Step]:
    if any(isinstance(argument, Row) for argument in arguments):
        raise Unsupported("passing a row to a call", context.where(rest))
    if isinstance(callee, Function):
        return inline(context, rest, callee, arguments, keywords)
    if (
        isinstance(callee, Opaque)
        and (callee.value is map or callee.value is filter)
        and not keywords
    ):
        return transformed(context, rest, callee.value is filter, arguments)
    special = (
        call_datetime(callee, arguments, keywords)
        or call_path(callee, arguments, keywords)
        or call_regex(callee, arguments, keywords, rest.path)
    )
    if special is not None:
        item, faults = special
        return emit(context, rest, item, faults)
    try:
        iterable = call_iterable(callee, arguments, keywords)
    except (TypeError, ValueError) as error:
        return [context.raised(rest, type(error))]
    if iterable is not None:
        item, faults = iterable
        return emit(context, rest, item, faults)
    if isinstance(callee, Opaque) and not keywords:
        if callee.value is str.maketrans:
            item, faults = make_translation_table(arguments)
            return emit(context, rest, item, faults)
        if callee.value is type and len(arguments) == 1:
            return [rest.push(runtime_type(arguments[0])).advance()]
        if callee.value is isinstance and len(arguments) == 2:
            test = type_test(runtime_type(arguments[0]), constant_value(arguments[1]), True)
            return [rest.push(Value(test)).advance()]
    node, faults = call(callee, arguments, keywords)
    return emit(context, rest, Value(node), faults)


def transformed(
    context: Context, state: State, filtering: bool, arguments: Sequence[Item]
) -> list[Step]:
    if len(arguments) != 2:
        raise Unsupported("map and filter require a function and one iterable")
    function, iterable = arguments
    iterator, pending = iterator_of(iterable)
    if not iterator.items and iterator.source is None:
        return emit(context, state, Generator(source_of(iterator)), pending)
    source, element, item = indexed(iterator, f"{context.code.co_qualname}:{state.index}:map")
    nested = replace(context, protected=(), caller=(context, state))
    start = state.under(TRUE)
    if isinstance(function, Function):
        outcomes = inline(nested, start, function, [item], {})
    else:
        value, faults = (
            truth(item)
            if filtering and isinstance(function, Value) and function.node == lit(None)
            else call(function, [item])
        )
        outcomes = emit(nested, start, Value(value), faults)
    cases: list[tuple[Node, Node]] = []
    failures: list[Node] = []
    for outcome in outcomes:
        if isinstance(outcome, Raise):
            failures.append(outcome.path)
        elif isinstance(outcome, State):
            result = outcome.stack[-1]
            if filtering:
                value, faults = truth(result)
                failures.extend(conjunction(outcome.path, fault.when) for fault in faults)
            else:
                value = node_of(result)
            cases.append((outcome.path, value))
    value = choose(cases) if cases else lit("")
    fault = reduce(disjunction, failures, FALSE)
    node = Comprehension(
        source,
        element,
        node_of(item) if filtering else value,
        (value if cases else FALSE) if filtering else TRUE,
        fault,
        iterator.requires_native,
    )
    if fault != FALSE:
        validate_fault_handlers(context)
    return emit(context, state, Generator(node), pending)
