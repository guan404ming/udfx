import dis

import pytest

from tests.frontend.streams import VERSION_IDS, VERSIONS, Version, traced
from udfx.errors import Unsupported
from udfx.frontend.decode import decoder
from udfx.frontend.decode.builder import Builder
from udfx.frontend.decode.graph import Jump
from udfx.frontend.decode.ops import NULL, Call
from udfx.frontend.decode.tables import resolve
from udfx.ir import FALSE, Col, Lit
from udfx.ops import apply
from udfx.types import FLOAT, INT, STR

from .test_opcodes import NUMBER, branch


def keyword_round(number: float) -> float:
    return round(number, ndigits=2)


@pytest.mark.parametrize("version", VERSIONS, ids=VERSION_IDS)
@pytest.mark.parametrize(
    ("conversion", "specifier"),
    [(0, None), (1, None), (2, None), (3, None), (0, ""), (0, "d"), (2, "s")],
)
def test_format_opcode_versions(
    monkeypatch: pytest.MonkeyPatch, version: Version, conversion: int, specifier: str | None
) -> None:
    def function(number: int) -> int:
        return number

    resume, load, returned = dis.get_instructions(function)
    load = load._replace(opname="LOAD_FAST_BORROW" if version >= (3, 14) else "LOAD_FAST")
    instructions = [resume, load]
    if version == (3, 12):
        if specifier is not None:
            instructions.append(load._replace(opname="LOAD_CONST", argval=specifier))
        instructions.append(
            load._replace(
                opname="FORMAT_VALUE", arg=conversion | (4 if specifier is not None else 0)
            )
        )
    else:
        if conversion:
            instructions.append(load._replace(opname="CONVERT_VALUE", arg=conversion))
        if specifier is not None:
            instructions.append(load._replace(opname="LOAD_CONST", argval=specifier))
        instructions.append(
            load._replace(opname="FORMAT_SIMPLE" if specifier is None else "FORMAT_WITH_SPEC")
        )
    instructions.append(returned)
    if conversion == 3:
        with pytest.raises(Unsupported, match="conversion !a"):
            traced(monkeypatch, function, instructions, version, {"number": INT})
        return
    program = traced(monkeypatch, function, instructions, version, {"number": INT})
    assert program.value == apply("str", NUMBER)
    assert program.fault == FALSE


@pytest.mark.parametrize("version", VERSIONS, ids=VERSION_IDS)
def test_keyword_opcode_versions(monkeypatch: pytest.MonkeyPatch, version: Version) -> None:
    template = next(dis.get_instructions(keyword_round))
    instructions = [
        template._replace(opname="RESUME", arg=0),
        template._replace(opname="LOAD_GLOBAL", arg=1, argval="round"),
        template._replace(
            opname="LOAD_FAST_BORROW" if version >= (3, 14) else "LOAD_FAST",
            arg=0,
            argval="number",
        ),
        template._replace(opname="LOAD_CONST", argval=2),
        template._replace(
            opname="KW_NAMES" if version == (3, 12) else "LOAD_CONST",
            arg=keyword_round.__code__.co_consts.index(("ndigits",)),
            argval=("ndigits",),
        ),
        template._replace(opname="CALL" if version == (3, 12) else "CALL_KW", arg=2),
        template._replace(opname="RETURN_VALUE", arg=None),
    ]
    program = traced(monkeypatch, keyword_round, instructions, version, {"number": FLOAT})
    assert program.value == apply("round", Col("number", FLOAT), Lit(2, INT))


@pytest.mark.parametrize("version", VERSIONS, ids=VERSION_IDS)
@pytest.mark.parametrize("keywords", [False, True])
def test_unpacked_call_versions(
    monkeypatch: pytest.MonkeyPatch, version: Version, keywords: bool
) -> None:
    template = next(dis.get_instructions(branch))
    instructions = [
        template._replace(opname="RESUME", arg=0),
        template._replace(opname="LOAD_GLOBAL", arg=1, argval="str"),
        template._replace(opname="LOAD_FAST", arg=0, argval="s"),
        template._replace(opname="BUILD_TUPLE", arg=1, argval=1),
    ]
    if keywords:
        instructions.append(template._replace(opname="BUILD_MAP", arg=0, argval=0))
    elif version >= (3, 14):
        instructions.append(template._replace(opname="PUSH_NULL", arg=None))
    instructions.extend(
        [
            template._replace(
                opname="CALL_FUNCTION_EX", arg=None if version >= (3, 14) else int(keywords)
            ),
            template._replace(opname="RETURN_VALUE", arg=None),
        ]
    )
    program = traced(monkeypatch, branch, instructions, version, {"s": STR})
    assert program.value == apply("str", Col("s", STR))


@pytest.mark.parametrize(
    ("version", "popped"), [(version, 2 if version == (3, 12) else 1) for version in VERSIONS]
)
def test_loop_cleanup_versions(version: Version, popped: int) -> None:
    template = next(dis.get_instructions(branch))
    builder = Builder(branch.__code__, (template,))
    builder.push(1, 2, 3)
    resolve(version)["END_FOR"](builder, template._replace(opname="END_FOR"))
    assert builder.stack == [1, 2, 3][: 3 - popped]


@pytest.mark.parametrize(
    ("version", "constant"),
    [((3, 14), value) for value in (3, 4)] + [((3, 15), value) for value in (2, 3, 4, 5, 6)],
)
def test_reduction_specialization_uses_call_path(version: Version, constant: int) -> None:
    template = next(dis.get_instructions(branch))
    instructions = tuple(
        item._replace(offset=2 * index)
        for index, item in enumerate(
            [
                template._replace(opname="LOAD_GLOBAL", arg=0, argval="any"),
                template._replace(opname="COPY", arg=1),
                template._replace(opname="LOAD_COMMON_CONSTANT", arg=constant),
                template._replace(opname="IS_OP", arg=0),
                template._replace(opname="POP_JUMP_IF_FALSE", argval=12),
                template._replace(opname="NOT_TAKEN"),
                template._replace(opname="RETURN_VALUE", arg=None),
            ]
        )
    )
    decoded = decoder.build(branch.__code__, instructions, [], resolve(version))
    entry = decoded.blocks[decoded.entry]
    assert isinstance(entry.exit, Jump)
    assert entry.exit.edge.target == 12
    assert [type(op).__name__ for op in entry.ops] == ["Global"]


def test_call_layout_resolves_null_in_every_version() -> None:
    template = next(dis.get_instructions(branch))
    for version in ((3, 12), (3, 13)):
        instructions = tuple(
            item._replace(offset=2 * index)
            for index, item in enumerate(
                [
                    template._replace(opname="LOAD_GLOBAL", arg=1, argval="str"),
                    template._replace(opname="LOAD_FAST", arg=0, argval="s"),
                    template._replace(opname="CALL", arg=1),
                    template._replace(opname="RETURN_VALUE", arg=None),
                ]
            )
        )
        decoded = decoder.build(branch.__code__, instructions, [], resolve(version))
        call = next(op for op in decoded.blocks[decoded.entry].ops if isinstance(op, Call))
        assert call.receiver is None
        assert NULL not in (call.callee, *call.arguments)
