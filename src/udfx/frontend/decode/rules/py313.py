from __future__ import annotations

import dis

from udfx.frontend.decode.builder import Builder, Rule
from udfx.frontend.decode.ops import (
    NULL,
    Cell,
    DecodeError,
    Format,
    Global,
    MakeFunction,
    Store,
)
from udfx.frontend.decode.rules.loads import load_local
from udfx.frontend.decode.rules.values import (
    CONVERSIONS,
    FUNCTION_FIELDS,
    call_with,
    function_code,
)


def end_for(builder: Builder, instruction: dis.Instruction) -> None:
    builder.carried -= 1
    builder.pop()


def push_null(builder: Builder, instruction: dis.Instruction) -> None:
    builder.carried += 1
    builder.push(NULL)


def load_global(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.produce(Global, instruction.argval))
    if instruction.arg and instruction.arg & 1:
        builder.push(NULL)


def load_fast(builder: Builder, instruction: dis.Instruction) -> None:
    if instruction.argval in builder.cells:
        builder.push(builder.produce(Cell, instruction.argval))
        return
    load_local(builder, instruction)


def load_pair(builder: Builder, instruction: dis.Instruction) -> None:
    builder.carried += 1
    for name in instruction.argval:
        load_fast(builder, instruction._replace(argval=name))


def store_pair(builder: Builder, instruction: dis.Instruction) -> None:
    builder.carried += 1
    for name in instruction.argval:
        builder.emit(Store, name, builder.pop())


def store_load(builder: Builder, instruction: dis.Instruction) -> None:
    builder.carried += 1
    head, tail = instruction.argval
    builder.emit(Store, head, builder.pop())
    load_fast(builder, instruction._replace(argval=tail))


def call_keywords(builder: Builder, instruction: dis.Instruction) -> None:
    names = builder.constants.get(builder.pop())
    if not isinstance(names, tuple):
        raise DecodeError(f"keyword names are not a constant at {builder.offset}")
    call_with(builder, instruction.arg or 0, names)


def convert_value(builder: Builder, instruction: dis.Instruction) -> None:
    source = builder.pop()
    conversion = CONVERSIONS[(instruction.arg or 0) & 3]
    builder.push(builder.produce(Format, source, conversion, None))


def format_simple(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.produce(Format, builder.pop(), None, None))


def format_with_spec(builder: Builder, instruction: dis.Instruction) -> None:
    specification, source = builder.pop(), builder.pop()
    builder.push(builder.produce(Format, source, None, specification))


def to_bool(builder: Builder, instruction: dis.Instruction) -> None:
    builder.carried += 1
    builder.push(builder.truth(builder.pop()))


def make_function(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.produce(MakeFunction, function_code(builder, builder.pop())))


def set_function_attribute(builder: Builder, instruction: dis.Instruction) -> None:
    builder.carried -= 1
    function, attribute = builder.pop(), builder.pop()
    field = FUNCTION_FIELDS[instruction.argval]
    if field != "annotations":
        builder.replace_operation(function, **{field: attribute})
    builder.push(function)


OVERRIDES: dict[str, Rule] = {
    "END_FOR": end_for,
    "PUSH_NULL": push_null,
    "LOAD_GLOBAL": load_global,
    "LOAD_FAST": load_fast,
    "LOAD_FAST_LOAD_FAST": load_pair,
    "STORE_FAST_LOAD_FAST": store_load,
    "STORE_FAST_STORE_FAST": store_pair,
    "CALL_KW": call_keywords,
    "CONVERT_VALUE": convert_value,
    "FORMAT_SIMPLE": format_simple,
    "FORMAT_WITH_SPEC": format_with_spec,
    "TO_BOOL": to_bool,
    "MAKE_FUNCTION": make_function,
    "SET_FUNCTION_ATTRIBUTE": set_function_attribute,
}
