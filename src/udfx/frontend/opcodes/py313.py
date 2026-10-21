from __future__ import annotations

import dis
from collections.abc import Iterable
from types import CodeType

PAIRS = {
    "LOAD_FAST_LOAD_FAST": ("LOAD_FAST", "LOAD_FAST"),
    "STORE_FAST_LOAD_FAST": ("STORE_FAST", "LOAD_FAST"),
    "STORE_FAST_STORE_FAST": ("STORE_FAST", "STORE_FAST"),
}


def normalize(code: CodeType) -> tuple[dis.Instruction, ...]:
    return lower(code, dis.get_instructions(code))


def lower(code: CodeType, instructions: Iterable[dis.Instruction]) -> tuple[dis.Instruction, ...]:
    cells = {*code.co_cellvars, *code.co_freevars}
    lowered: list[dis.Instruction] = []
    for instruction in (part for whole in instructions for part in split(whole)):
        match instruction.opname:
            case "EXTENDED_ARG":
                lowered.append(instruction._replace(opname="NOP"))
            case "POP_TOP" if lowered and lowered[-1].opname == "END_FOR":
                lowered.append(instruction._replace(opname="NOP"))
            case "GET_ITER" if [part.opname for part in lowered[-2:]] == ["BUILD_LIST", "SWAP"]:
                lowered.append(instruction._replace(opname="NOP"))
            case "CONVERT_VALUE" | "FORMAT_SIMPLE" | "FORMAT_WITH_SPEC":
                flags = (
                    instruction.arg
                    if instruction.opname == "CONVERT_VALUE"
                    else (4 if instruction.opname == "FORMAT_WITH_SPEC" else 0)
                )
                lowered.append(instruction._replace(opname="FORMAT_VALUE", arg=flags))
            case "TO_BOOL":
                negate = instruction._replace(opname="UNARY_NOT", arg=None, argval=None, argrepr="")
                lowered.extend((negate, negate._replace(offset=instruction.offset + 1)))
            case "PUSH_NULL":
                swap = instruction._replace(
                    opname="SWAP", arg=2, argval=2, argrepr="2", offset=instruction.offset + 1
                )
                lowered.extend((instruction, swap))
            case "SET_FUNCTION_ATTRIBUTE":
                function = lowered.pop()
                flags = (function.arg or 0) | instruction.argval
                lowered.append(function._replace(arg=flags, argval=flags))
            case "CALL_KW":
                lowered.append(lowered.pop()._replace(opname="KW_NAMES"))
                lowered.append(instruction._replace(opname="CALL"))
            case "LOAD_FAST" if instruction.argval in cells:
                lowered.append(instruction._replace(opname="LOAD_CLOSURE"))
            case _:
                lowered.append(instruction)
    return tuple(lowered)


def split(instruction: dis.Instruction) -> tuple[dis.Instruction, ...]:
    if instruction.opname not in PAIRS:
        return (instruction,)
    (first, second), (head, tail) = PAIRS[instruction.opname], instruction.argval
    argument = instruction.arg or 0
    return (
        instruction._replace(opname=first, arg=argument >> 4, argval=head, argrepr=head),
        instruction._replace(
            opname=second,
            arg=argument & 15,
            argval=tail,
            argrepr=tail,
            offset=instruction.offset + 1,
        ),
    )
