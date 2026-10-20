from __future__ import annotations

import dis
import sys
from collections.abc import Callable
from types import CodeType

from udfx.frontend.opcodes import py312, py313, py314

NORMALIZERS: dict[tuple[int, int], Callable[[CodeType], tuple[dis.Instruction, ...]]] = {
    (3, 12): py312.normalize,
    (3, 13): py313.normalize,
    (3, 14): py314.normalize,
}


def normalize(code: CodeType) -> tuple[dis.Instruction, ...]:
    return NORMALIZERS[sys.version_info.major, sys.version_info.minor](code)
