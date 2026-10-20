from __future__ import annotations

import dis
from types import CodeType


def normalize(code: CodeType) -> tuple[dis.Instruction, ...]:
    return tuple(dis.get_instructions(code))
