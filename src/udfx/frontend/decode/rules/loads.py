from __future__ import annotations

import dis

from udfx.frontend.decode.builder import Builder
from udfx.frontend.decode.graph import Refuse
from udfx.frontend.decode.ops import (
    NULL,
    Attribute,
    Cell,
    Delete,
    Global,
    Import,
    Local,
    Store,
)
from udfx.frontend.decode.refusals import FEATURES


def skip(builder: Builder, instruction: dis.Instruction) -> None:
    return None


def refuse(builder: Builder, instruction: dis.Instruction) -> None:
    reason = FEATURES.get(instruction.opname, f"opcode {instruction.opname}")
    builder.finish(Refuse, reason)


def load_local(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.produce(Local, instruction.argval))


def load_clear(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.produce(Local, instruction.argval, clear=True))


def load_cell(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.produce(Local, instruction.argval, cell=True))


def load_closure(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.produce(Cell, instruction.argval))


def load_constant(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.constant(instruction.argval))


def load_global(builder: Builder, instruction: dis.Instruction) -> None:
    if instruction.arg and instruction.arg & 1:
        builder.push(NULL)
    builder.push(builder.produce(Global, instruction.argval))


def push_null(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(NULL)


def import_name(builder: Builder, instruction: dis.Instruction) -> None:
    level, names = builder.pop_many(2)
    builder.push(builder.produce(Import, instruction.argval, level, names))


def import_from(builder: Builder, instruction: dis.Instruction) -> None:
    builder.reach(1)
    result = builder.new()
    builder.emit(Attribute, (result,), builder.stack[-1], instruction.argval, False)
    builder.push(result)


def load_attribute(builder: Builder, instruction: dis.Instruction) -> None:
    owner = builder.pop()
    if instruction.arg and instruction.arg & 1:
        results = (builder.new(), builder.new())
        builder.emit(Attribute, results, owner, instruction.argval, True)
        builder.push(*results)
        return
    result = builder.new()
    builder.emit(Attribute, (result,), owner, instruction.argval, False)
    builder.push(result)


def store_local(builder: Builder, instruction: dis.Instruction) -> None:
    builder.emit(Store, instruction.argval, builder.pop())


def store_cell(builder: Builder, instruction: dis.Instruction) -> None:
    builder.emit(Store, instruction.argval, builder.pop(), cell=True)


def delete_local(builder: Builder, instruction: dis.Instruction) -> None:
    builder.emit(Delete, instruction.argval)


def pop_top(builder: Builder, instruction: dis.Instruction) -> None:
    builder.pop()


def copy(builder: Builder, instruction: dis.Instruction) -> None:
    builder.reach(instruction.arg or 1)
    builder.push(builder.stack[-(instruction.arg or 1)])


def swap(builder: Builder, instruction: dis.Instruction) -> None:
    depth = instruction.arg or 2
    builder.reach(depth)
    builder.stack[-1], builder.stack[-depth] = builder.stack[-depth], builder.stack[-1]
