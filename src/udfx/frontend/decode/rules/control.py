from __future__ import annotations

import dis

from udfx.frontend.decode.builder import Builder
from udfx.frontend.decode.graph import Branch, Fail, ForEach, Jump, Return
from udfx.frontend.decode.ops import (
    Compare,
    DecodeError,
    ExceptionMatch,
    GeneratorStart,
    Placeholder,
    Yield,
)
from udfx.frontend.decode.rules.loads import refuse


def return_value(builder: Builder, instruction: dis.Instruction) -> None:
    source = builder.pop()
    if builder.stack and not builder.lenient:
        raise DecodeError(f"stack not empty at return {builder.offset}")
    builder.finish(Return, source)


def return_constant(builder: Builder, instruction: dis.Instruction) -> None:
    builder.push(builder.constant(instruction.argval))
    return_value(builder, instruction)


def branch_if_false(builder: Builder, instruction: dis.Instruction) -> None:
    condition = builder.truth(builder.pop())
    branch(builder, instruction, condition)


def branch_if_true(builder: Builder, instruction: dis.Instruction) -> None:
    condition = builder.negation(builder.pop())
    branch(builder, instruction, condition)


def branch_if_none(builder: Builder, instruction: dis.Instruction) -> None:
    source = builder.pop()
    condition = builder.produce(Compare, "is not", source, builder.constant(None))
    branch(builder, instruction, condition)


def branch_if_not_none(builder: Builder, instruction: dis.Instruction) -> None:
    source = builder.pop()
    condition = builder.produce(Compare, "is", source, builder.constant(None))
    branch(builder, instruction, condition)


def branch(builder: Builder, instruction: dis.Instruction, condition: int) -> None:
    then = builder.edge(builder.instructions[builder.index + 1].offset)
    builder.finish(Branch, condition, then, builder.edge(instruction.argval))


def jump(builder: Builder, instruction: dis.Instruction) -> None:
    builder.finish(Jump, builder.edge(instruction.argval))


def jump_backward(builder: Builder, instruction: dis.Instruction) -> None:
    if builder.instructions[builder.index_of[instruction.argval]].opname == "FOR_ITER":
        jump(builder, instruction)
        return
    refuse(builder, instruction)


def for_each(builder: Builder, instruction: dis.Instruction, width: int = 1) -> None:
    iterator, *extra = builder.pop_many(width)
    accumulator = builder.stack[-1] if builder.stack else None
    item, advanced, dummy = builder.new(), builder.new(), builder.new()
    saved = list(builder.stack)
    builder.push(advanced, *extra, item)
    body = builder.edge(builder.instructions[builder.index + 1].offset)
    builder.stack[:] = saved
    builder.push(iterator, *extra, dummy)
    done = builder.edge(instruction.argval)
    builder.finish(
        ForEach, iterator, item, advanced, dummy, accumulator, body, done, instruction.argval
    )


def end_for(builder: Builder, instruction: dis.Instruction) -> None:
    builder.carried -= 1
    builder.pop_many(2)


def push_exception(builder: Builder, instruction: dis.Instruction) -> None:
    exception = builder.pop()
    builder.push(builder.produce(Placeholder), exception)


def check_exception(builder: Builder, instruction: dis.Instruction) -> None:
    classes = builder.pop()
    builder.reach(1)
    builder.push(builder.produce(ExceptionMatch, builder.stack[-1], classes))


def reraise(builder: Builder, instruction: dis.Instruction) -> None:
    exception = builder.pop()
    builder.stack.clear()
    builder.finish(Fail, exception)


def yield_value(builder: Builder, instruction: dis.Instruction) -> None:
    value, result = builder.pop(), builder.new()
    builder.emit(Yield, value, result)
    builder.push(result)


def return_generator(builder: Builder, instruction: dis.Instruction) -> None:
    builder.emit(GeneratorStart)
    builder.push(builder.produce(Placeholder))
