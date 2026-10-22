# pyright: basic
from __future__ import annotations

import dis
import types
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from tests.corpus.cases import CASES
from udfx.frontend.decode import DecodeError, decode
from udfx.frontend.decode.builder import Builder
from udfx.frontend.decode.decoder import attach_handler
from udfx.frontend.decode.graph import Branch, Fail, ForEach, Refuse
from udfx.frontend.decode.ops import (
    NULL,
    Append,
    Attribute,
    BinaryOp,
    Boolean,
    Call,
    Format,
    Store,
    Yield,
)
from udfx.frontend.decode.rules.control import return_value


def method_call(text):
    return text.strip() + "a"


def plain_call(text):
    return len(text)


def keyword_call(text):
    return round(text, ndigits=2)


def conditional(flag, left, right):
    return left if flag else right


def comprehension(items):
    return [item.strip() for item in items if item]


def generator(items):
    return ",".join(item.lower() for item in items)


def handled(value):
    try:
        return int(value)
    except ValueError:
        return -1


def finalized(value):
    try:
        result = int(value)
    finally:
        value = 1
    return result


def formatted(value):
    return f"{value!r:>5}-{value}"


def inverted(value):
    return ~value


def waiting(value):
    with value:
        return 1


def spinning(value):
    while value:
        value = value - 1
    return value


def blocks(function: Callable[..., object]):
    return decode(function.__code__).blocks.values()


def operations(function: Callable[..., object]):
    return [op for block in blocks(function) for op in block.ops]


def exits(function: Callable[..., object]):
    return [block.exit for block in blocks(function)]


def codes(code: types.CodeType):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from codes(constant)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_corpus_functions_decode(case) -> None:
    for code in codes(case.function.__code__):
        assert decode(code).blocks


def test_method_call_has_a_receiver() -> None:
    attribute = next(op for op in operations(method_call) if isinstance(op, Attribute))
    call = next(op for op in operations(method_call) if isinstance(op, Call))
    assert attribute.method
    assert (call.callee, call.receiver) == attribute.results


def test_plain_call_resolves_null_in_the_decoder() -> None:
    call = next(op for op in operations(plain_call) if isinstance(op, Call))
    assert call.receiver is None
    assert len(call.arguments) == 1


def test_keyword_names_resolve_in_the_decoder() -> None:
    call = next(op for op in operations(keyword_call) if isinstance(op, Call))
    assert [name for name, _ in call.keywords] == ["ndigits"]
    assert len(call.arguments) == 1


def test_branch_reads_a_bool() -> None:
    branch = next(exit for exit in exits(conditional) if isinstance(exit, Branch))
    assert any(
        isinstance(op, Boolean) and op.result == branch.condition for op in operations(conditional)
    )


def test_comprehension_is_a_foreach_region() -> None:
    loop = next(exit for exit in exits(comprehension) if isinstance(exit, ForEach))
    assert loop.accumulator is not None
    assert any(isinstance(op, Append) for op in operations(comprehension))


def test_comprehension_cleanup_is_transparent() -> None:
    assert all(op.handler is None for op in operations(comprehension))
    assert not any(isinstance(exit, Fail) for exit in exits(comprehension))


def test_unreachable_code_is_still_scanned() -> None:
    decoded = decode(comprehension.__code__)
    assert any(isinstance(block.exit, Fail) for block in decoded.orphans.values())
    assert any(isinstance(op, Fail) for op in decoded.operations(0))
    assert not any(isinstance(block.exit, Fail) for block in decoded.blocks.values())


def test_generator_expression_yields() -> None:
    code = next(code for code in codes(generator.__code__) if code.co_name == "<genexpr>")
    decoded = decode(code)
    assert any(isinstance(op, Yield) for block in decoded.blocks.values() for op in block.ops)
    assert all(op.handler is None for block in decoded.blocks.values() for op in block.ops)


def test_exception_handler_carries_the_table_depth() -> None:
    handlers = {op.handler for op in operations(handled) if op.handler is not None}
    assert handlers
    for handler in handlers:
        block = decode(handled.__code__).blocks[handler.edge.target]
        assert len(block.parameters) == len(handler.edge.arguments)
        assert handler.edge.arguments[-1] == handler.exception


def test_finally_keeps_both_copies_of_the_body() -> None:
    stores = [op for op in operations(finalized) if isinstance(op, Store) and op.name == "value"]
    assert len(stores) == 2
    assert any(isinstance(exit, Fail) for exit in exits(finalized))


def test_format_names_conversion_and_specification() -> None:
    formats = [op for op in operations(formatted) if isinstance(op, Format)]
    assert any(op.conversion is not None for op in formats)
    assert any(op.specification is not None for op in formats)


def test_binary_operator_keeps_its_symbol() -> None:
    assert [op.symbol for op in operations(method_call) if isinstance(op, BinaryOp)] == ["+"]


@pytest.mark.parametrize(
    ("function", "reason"),
    [
        (inverted, "opcode UNARY_INVERT"),
        (waiting, "with blocks"),
        (spinning, "loops and comprehensions"),
    ],
)
def test_unsupported_opcode_ends_its_block(function, reason) -> None:
    refusals = [exit for exit in exits(function) if isinstance(exit, Refuse)]
    assert [refusal.reason for refusal in refusals] == [reason]


def test_every_value_is_defined_before_use() -> None:
    for code in codes(comprehension.__code__):
        for block in decode(code).blocks.values():
            defined = set(block.parameters) | {NULL}
            for op in (*block.ops, block.exit):
                for field in ("source", "left", "right", "owner", "callee", "container"):
                    used = getattr(op, field, None)
                    if isinstance(used, int):
                        assert used in defined
                for field in ("result", "item", "advanced"):
                    value = getattr(op, field, None)
                    if isinstance(value, int):
                        defined.add(value)
                defined.update(getattr(op, "results", ()))


def instructions_of(function: Callable[..., object]) -> tuple[dis.Instruction, ...]:
    return tuple(dis.get_instructions(function.__code__))


def test_merge_with_a_different_depth_raises() -> None:
    builder = Builder(plain_call.__code__, instructions_of(plain_call))
    builder.claim(10, 1)
    with pytest.raises(DecodeError):
        builder.claim(10, 2)


def test_return_with_a_stack_raises() -> None:
    builder = Builder(plain_call.__code__, instructions_of(plain_call))
    builder.push(1, 2)
    with pytest.raises(DecodeError):
        return_value(builder, instructions_of(plain_call)[0])


def test_handler_deeper_than_the_stack_raises() -> None:
    builder = Builder(plain_call.__code__, instructions_of(plain_call))
    builder.offset = 5
    entry = SimpleNamespace(
        start=0, end=10, target=instructions_of(plain_call)[0].offset, depth=2, lasti=False
    )
    with pytest.raises(DecodeError):
        attach_handler(builder, [entry])


def test_underflow_raises() -> None:
    builder = Builder(plain_call.__code__, instructions_of(plain_call))
    with pytest.raises(DecodeError):
        builder.pop()
