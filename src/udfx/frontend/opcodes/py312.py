from __future__ import annotations

import dis
from types import CodeType


def normalize(code: CodeType) -> tuple[dis.Instruction, ...]:
    return tuple(
        instruction._replace(argval=code.co_consts[instruction.arg or 0])
        if instruction.opname == "KW_NAMES"
        else instruction._replace(opname="NOP")
        if instruction.opname == "EXTENDED_ARG"
        else instruction
        for instruction in dis.get_instructions(code)
    )
