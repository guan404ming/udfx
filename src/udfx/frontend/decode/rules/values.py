from __future__ import annotations

import dis
from types import CodeType

from udfx.frontend.decode.builder import Builder, Rule
from udfx.frontend.decode.ops import (
    Append,
    BinaryOp,
    Build,
    Call,
    Compare,
    Cut,
    DecodeError,
    Extend,
    Format,
    Iterate,
    MakeFunction,
    MapAdd,
    Merge,
    Negate,
    Slice,
    Subscript,
    Unpack,
    UnpackedCall,
)

CONVERSIONS = (None, "s", "r", "a")
FUNCTION_FIELDS = {1: "defaults", 2: "keyword_defaults", 4: "annotations", 8: "closure"}


def keyword_names(builder: Builder, instruction: dis.Instruction) -> None:
    builder.keywords = tuple(instruction.argval)


def call_with(builder: Builder, count: int, names: tuple[str, ...]) -> None:
    arguments = builder.pop_many(count)
    callee, receiver = builder.slots(*builder.pop_many(2))
    split = len(arguments) - len(names)
    keywords = tuple(zip(names, arguments[split:], strict=True))
    builder.push(builder.produce(Call, callee, arguments[:split], keywords, receiver))


def call(builder: Builder, instruction: dis.Instruction) -> None:
    names, builder.keywords = builder.keywords, ()
    call_with(builder, instruction.arg or 0, names)


def call_unpacked(builder: Builder, instruction: dis.Instruction) -> None:
    keywords = builder.pop() if instruction.arg and instruction.arg & 1 else None
    arguments = builder.pop()
    callee, receiver = builder.slots(*builder.pop_many(2))
    builder.push(builder.produce(UnpackedCall, callee, receiver, arguments, keywords))


def merge_keywords(builder: Builder, instruction: dis.Instruction) -> None:
    builder.emit(Merge, builder.pop())


def binary_op(builder: Builder, instruction: dis.Instruction) -> None:
    binary(builder, BinaryOp, instruction.argrepr)


def compare(builder: Builder, instruction: dis.Instruction) -> None:
    binary(builder, Compare, str(instruction.argval))


def identity(builder: Builder, instruction: dis.Instruction) -> None:
    binary(builder, Compare, "is not" if instruction.arg else "is")


def membership(builder: Builder, instruction: dis.Instruction) -> None:
    binary(builder, Compare, "not in" if instruction.arg else "in")


def binary(builder: Builder, kind: type[BinaryOp | Compare], symbol: str) -> None:
    builder.push(builder.produce(kind, symbol, *builder.pop_many(2)))


def unary_not(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.negation(builder.pop()))


def unary_negative(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.produce(Negate, builder.pop()))


def subscript(builder: Builder, instruction: dis.Instruction) -> None:
    index, container = builder.pop(), builder.pop()
    builder.push(builder.produce(Subscript, container, index))


def cut(builder: Builder, instruction: dis.Instruction) -> None:
    stop, start, container = builder.pop(), builder.pop(), builder.pop()
    builder.push(builder.produce(Cut, container, start, stop))


def build_slice(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.produce(Slice, builder.pop_many(instruction.argval)))


def build(kind: str, width: int = 1, extra: int = 0) -> Rule:
    def rule(builder: Builder, instruction: dis.Instruction) -> None:
        items = builder.pop_many(width * instruction.argval + extra)
        builder.push(builder.produce(Build, kind, items))

    return rule


def extend(kind: str) -> Rule:
    def rule(builder: Builder, instruction: dis.Instruction) -> None:
        source = builder.pop()
        builder.reach(instruction.argval)
        position = len(builder.stack) - instruction.argval
        result = builder.produce(Extend, kind, builder.stack[position], source)
        builder.stack[position] = result

    return rule


def list_append(builder: Builder, instruction: dis.Instruction) -> None:
    builder.emit(Append, builder.pop())


def map_add(builder: Builder, instruction: dis.Instruction) -> None:
    key, value = builder.pop_many(2)
    builder.reach(instruction.argval)
    position = len(builder.stack) - instruction.argval
    builder.stack[position] = builder.produce(MapAdd, builder.stack[position], key, value)


def unpack(builder: Builder, instruction: dis.Instruction) -> None:
    starred = instruction.opname == "UNPACK_EX"
    before = instruction.argval & 255 if starred else instruction.argval
    after = instruction.argval >> 8 if starred else 0
    size = before + after + int(starred)
    source = builder.pop()
    results = tuple(builder.new() for _ in range(size))
    builder.emit(Unpack, results, source, before, after, starred)
    builder.push(*reversed(results))


def get_iterator(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.produce(Iterate, builder.pop()))


def format_value(builder: Builder, instruction: dis.Instruction) -> None:
    flags = instruction.arg or 0
    specification = builder.pop() if flags & 4 else None
    source = builder.pop()
    builder.push(builder.produce(Format, source, CONVERSIONS[flags & 3], specification))


def function_code(builder: Builder, identifier: int) -> CodeType:
    code = builder.constants.get(identifier)
    if not isinstance(code, CodeType):
        raise DecodeError(f"function code is not a constant at {builder.offset}")
    return code


def make_function(builder: Builder, instruction: dis.Instruction) -> None:
    flags = instruction.arg or 0
    code = function_code(builder, builder.pop())
    fields = {
        name: builder.pop()
        for bit, name in sorted(FUNCTION_FIELDS.items(), reverse=True)
        if flags & bit
    }
    fields.pop("annotations", None)
    builder.push(builder.produce(MakeFunction, code, **fields))
