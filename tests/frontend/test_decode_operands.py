import dis
from types import SimpleNamespace

import pytest

from tests.frontend.streams import VERSION_IDS, VERSIONS, Version
from udfx.frontend import trace
from udfx.frontend.decode.builder import Builder
from udfx.frontend.decode.decoder import attach_handler
from udfx.frontend.decode.ops import Call, Constant, Global, Unpack
from udfx.frontend.decode.tables import resolve
from udfx.ir import Col, If, lit
from udfx.ops import apply
from udfx.types import BOOL, INT, STR


def conditional_call(text: str, flag: bool) -> int:
    return len(text if flag else "abc")


@pytest.mark.parametrize("version", VERSIONS, ids=VERSION_IDS)
def test_call_sentinel_resolves_after_a_block_edge(version: Version) -> None:
    instructions = tuple(dis.get_instructions(conditional_call))
    builder = Builder(conditional_call.__code__, instructions)
    table = resolve(version)
    table["LOAD_GLOBAL"](builder, instructions[0]._replace(arg=1, argval="len"))
    target = instructions[-1].offset
    edge = builder.edge(target)
    builder.stack = list(builder.claims[target])
    table["CALL"](builder, instructions[0]._replace(arg=0))
    call = builder.ops[-1]
    assert isinstance(call, Call)
    assert call.receiver is None
    index = 1 if version == (3, 12) else 0
    assert call.callee == builder.claims[target][index]
    loaded = builder.ops[0]
    assert isinstance(loaded, Global)
    assert edge.arguments[index] == loaded.result


def test_call_with_a_conditional_argument_keeps_its_callee() -> None:
    program = trace(conditional_call, {"text": STR, "flag": BOOL})
    assert program.value == If(Col("flag", BOOL), apply("len", Col("text", STR)), lit(3), INT)


@pytest.mark.parametrize("version", VERSIONS, ids=VERSION_IDS)
@pytest.mark.parametrize(
    ("before", "after", "starred"), [(2, 0, False), (1, 2, True), (0, 0, True)]
)
def test_unpack_exposes_counts_and_source_order(
    version: Version, before: int, after: int, starred: bool
) -> None:
    instructions = tuple(dis.get_instructions(conditional_call))
    builder = Builder(conditional_call.__code__, instructions)
    source = builder.constant("items")
    builder.push(source)
    opcode = "UNPACK_EX" if starred else "UNPACK_SEQUENCE"
    argument = before | (after << 8) if starred else before
    resolve(version)[opcode](
        builder, instructions[0]._replace(opname=opcode, arg=argument, argval=argument)
    )
    unpack = builder.ops[-1]
    assert isinstance(unpack, Unpack)
    assert (unpack.source, unpack.before, unpack.after, unpack.starred) == (
        source,
        before,
        after,
        starred,
    )
    assert len(unpack.results) == before + after + int(starred)
    assert tuple(builder.pop() for _ in unpack.results) == unpack.results


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("lasti", [False, True])
def test_handler_exposes_retained_values_and_exception(depth: int, lasti: bool) -> None:
    instructions = tuple(dis.get_instructions(conditional_call))
    builder = Builder(conditional_call.__code__, instructions)
    retained = tuple(builder.constant(value) for value in range(depth))
    builder.push(*retained)
    builder.offset = 5
    builder.carried = 3
    entry = SimpleNamespace(
        start=0, end=10, target=instructions[0].offset, depth=depth, lasti=lasti
    )
    attach_handler(builder, [entry])
    handler = builder.handler
    assert handler is not None
    assert len(builder.claims[entry.target]) == depth + int(lasti) + 1
    assert handler.edge.target == entry.target
    assert handler.edge.arguments[:depth] == retained
    assert handler.edge.arguments[-1] == handler.exception
    assert len(handler.edge.arguments) == depth + int(lasti) + 1
    assert builder.carried == 3
    if lasti:
        position = builder.ops[-1]
        assert isinstance(position, Constant)
        assert (position.result, position.value) == (handler.edge.arguments[-2], builder.offset)
        assert position.weight == 0
