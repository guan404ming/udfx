from __future__ import annotations

import dis

from udfx.frontend.decode.builder import Builder, Rule
from udfx.frontend.decode.ops import NULL
from udfx.frontend.decode.rules import control, loads, py314, values


def load_common_constant(builder: Builder, instruction: dis.Instruction) -> None:
    if instruction.arg is not None and instruction.arg >= 7:
        loads.load_constant(builder, instruction)
    else:
        loads.refuse(builder, instruction)


def get_iterator(builder: Builder, instruction: dis.Instruction) -> None:
    values.get_iterator(builder, instruction)
    builder.push(NULL)


def for_each(builder: Builder, instruction: dis.Instruction) -> None:
    control.for_each(builder, instruction, width=2)


def pop_iterator(builder: Builder, instruction: dis.Instruction) -> None:
    builder.pop_many(2)


def copy(builder: Builder, instruction: dis.Instruction) -> None:
    py314.copy(builder, instruction, constants=(2, 3, 4, 5, 6))


OVERRIDES: dict[str, Rule] = {
    **py314.OVERRIDES,
    "LOAD_COMMON_CONSTANT": load_common_constant,
    "GET_ITER": get_iterator,
    "FOR_ITER": for_each,
    "POP_ITER": pop_iterator,
    "COPY": copy,
}
