from __future__ import annotations

import dis

from udfx.frontend.decode.builder import Builder, Rule
from udfx.frontend.decode.graph import Jump
from udfx.frontend.decode.ops import NULL, Cut, Subscript, UnpackedCall
from udfx.frontend.decode.rules import loads, py313, values

ANNOTATE = 0x10
ANNOTATIONS = 0x04
REDUCTIONS = ("LOAD_GLOBAL", "COPY", "LOAD_COMMON_CONSTANT", "IS_OP", "POP_JUMP_IF_FALSE")


def load_small_int(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.constant(instruction.arg))


def binary_op(builder: Builder, instruction: dis.Instruction) -> None:
    if instruction.argrepr != "[]":
        values.binary_op(builder, instruction)
        return
    index, container = builder.pop(), builder.pop()
    bounds = builder.constants.get(index)
    if not isinstance(bounds, slice) or bounds.step is not None:
        builder.push(builder.produce(Subscript, container, index))
        return
    builder.retract(index)
    builder.carried += 1
    start, stop = builder.constant(bounds.start), builder.constant(bounds.stop)
    builder.push(builder.produce(Cut, container, start, stop))


def set_function_attribute(builder: Builder, instruction: dis.Instruction) -> None:
    flag = ANNOTATIONS if instruction.argval == ANNOTATE else instruction.argval
    py313.set_function_attribute(builder, instruction._replace(argval=flag))


def call_unpacked(builder: Builder, instruction: dis.Instruction) -> None:
    keywords = builder.pop()
    arguments = builder.pop()
    callee, receiver = builder.slots(*builder.pop_many(2))
    result = builder.produce(
        UnpackedCall, callee, receiver, arguments, None if keywords == NULL else keywords
    )
    builder.push(result)


def copy(
    builder: Builder, instruction: dis.Instruction, constants: tuple[int, ...] = (3, 4)
) -> None:
    window = builder.instructions[builder.index - 1 : builder.index + 4]
    if (
        builder.index > 0
        and tuple(part.opname for part in window) == REDUCTIONS
        and window[2].arg in constants
    ):
        builder.carried += 3
        builder.finish(Jump, builder.edge(window[4].argval))
        return
    loads.copy(builder, instruction)


OVERRIDES: dict[str, Rule] = {
    **py313.OVERRIDES,
    "LOAD_FAST_BORROW": py313.load_fast,
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW": py313.load_pair,
    "LOAD_SMALL_INT": load_small_int,
    "NOT_TAKEN": loads.skip,
    "POP_ITER": loads.pop_top,
    "BINARY_OP": binary_op,
    "SET_FUNCTION_ATTRIBUTE": set_function_attribute,
    "CALL_FUNCTION_EX": call_unpacked,
    "COPY": copy,
    "LOAD_COMMON_CONSTANT": loads.refuse,
}
