from __future__ import annotations

import dis
from collections.abc import Iterable
from types import CodeType

from udfx.frontend.opcodes import py313

RENAMES = {
    "LOAD_FAST_BORROW": "LOAD_FAST",
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW": "LOAD_FAST_LOAD_FAST",
    "LOAD_SMALL_INT": "LOAD_CONST",
    "NOT_TAKEN": "NOP",
}

ANNOTATE = 0x10
ANNOTATIONS = 0x04


def normalize(code: CodeType) -> tuple[dis.Instruction, ...]:
    return py313.lower(code, lower(dis.get_instructions(code)))


def lower(instructions: Iterable[dis.Instruction]) -> list[dis.Instruction]:
    lowered: list[dis.Instruction] = []
    for instruction in instructions:
        match instruction.opname:
            case name if name in RENAMES:
                lowered.append(instruction._replace(opname=RENAMES[name]))
            case "BINARY_OP" if instruction.argrepr == "[]":
                lowered.extend(subscript(lowered.pop(), instruction))
            case "SET_FUNCTION_ATTRIBUTE" if instruction.argval == ANNOTATE:
                lowered.append(instruction._replace(arg=ANNOTATIONS, argval=ANNOTATIONS))
            case _:
                lowered.append(instruction)
    return lowered


def subscript(index: dis.Instruction, instruction: dis.Instruction) -> tuple[dis.Instruction, ...]:
    bounds = index.argval
    if index.opname != "LOAD_CONST" or not isinstance(bounds, slice) or bounds.step is not None:
        return index, instruction._replace(
            opname="BINARY_SUBSCR", arg=None, argval=None, argrepr=""
        )
    return (
        index._replace(argval=bounds.start, argrepr=repr(bounds.start)),
        index._replace(argval=bounds.stop, argrepr=repr(bounds.stop), offset=index.offset + 1),
        instruction._replace(opname="BINARY_SLICE", arg=None, argval=None, argrepr=""),
    )
