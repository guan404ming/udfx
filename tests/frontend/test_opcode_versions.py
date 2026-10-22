import dis
from types import ModuleType

import pytest

from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.opcodes import py312, py313, py314
from udfx.ir import FALSE, Col, Lit
from udfx.ops import apply
from udfx.types import FLOAT, INT, STR

from .test_opcodes import NUMBER, branch


def keyword_round(number: float) -> float:
    return round(number, ndigits=2)


@pytest.mark.parametrize("normalizer", [py312, py313, py314], ids=["3.12", "3.13", "3.14"])
@pytest.mark.parametrize(
    ("conversion", "specifier"),
    [(0, None), (1, None), (2, None), (3, None), (0, ""), (0, "d"), (2, "s")],
)
def test_format_opcode_versions(
    monkeypatch: pytest.MonkeyPatch,
    normalizer: ModuleType,
    conversion: int,
    specifier: str | None,
) -> None:
    def function(number: int) -> int:
        return number

    resume, load, returned = dis.get_instructions(function)
    load = load._replace(opname="LOAD_FAST_BORROW" if normalizer is py314 else "LOAD_FAST")
    instructions = [resume, load]
    if normalizer is py312:
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
    instructions = [item._replace(offset=2 * index) for index, item in enumerate(instructions)]
    monkeypatch.setattr(dis, "get_instructions", lambda code: iter(instructions))
    monkeypatch.setattr("udfx.frontend.context.normalize", normalizer.normalize)
    if conversion == 3:
        with pytest.raises(Unsupported, match="conversion !a"):
            trace(function, {"number": INT})
        return
    program = trace(function, {"number": INT})
    assert program.value == apply("str", NUMBER)
    assert program.fault == FALSE


@pytest.mark.parametrize("normalizer", [py312, py313, py314], ids=["3.12", "3.13", "3.14"])
def test_keyword_opcode_versions(monkeypatch: pytest.MonkeyPatch, normalizer: ModuleType) -> None:
    template = next(dis.get_instructions(keyword_round))
    instructions = [
        template._replace(opname="RESUME", arg=0),
        template._replace(opname="LOAD_GLOBAL", arg=1, argval="round"),
        template._replace(
            opname="LOAD_FAST_BORROW" if normalizer is py314 else "LOAD_FAST",
            arg=0,
            argval="number",
        ),
        template._replace(opname="LOAD_CONST", argval=2),
        template._replace(
            opname="KW_NAMES" if normalizer is py312 else "LOAD_CONST",
            arg=keyword_round.__code__.co_consts.index(("ndigits",)),
            argval=("ndigits",),
        ),
        template._replace(opname="CALL" if normalizer is py312 else "CALL_KW", arg=2),
        template._replace(opname="RETURN_VALUE", arg=None),
    ]
    instructions = [item._replace(offset=2 * index) for index, item in enumerate(instructions)]
    monkeypatch.setattr(dis, "get_instructions", lambda code: iter(instructions))
    monkeypatch.setattr("udfx.frontend.context.normalize", normalizer.normalize)
    assert trace(keyword_round, {"number": FLOAT}).value == apply(
        "round", Col("number", FLOAT), Lit(2, INT)
    )


@pytest.mark.parametrize("normalizer", [py312, py313, py314], ids=["3.12", "3.13", "3.14"])
@pytest.mark.parametrize("keywords", [False, True])
def test_unpacked_call_versions(
    monkeypatch: pytest.MonkeyPatch, normalizer: ModuleType, keywords: bool
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
    elif normalizer is py314:
        instructions.append(template._replace(opname="PUSH_NULL", arg=None))
    instructions.extend(
        [
            template._replace(
                opname="CALL_FUNCTION_EX", arg=None if normalizer is py314 else int(keywords)
            ),
            template._replace(opname="RETURN_VALUE", arg=None),
        ]
    )
    instructions = [item._replace(offset=2 * index) for index, item in enumerate(instructions)]
    monkeypatch.setattr(dis, "get_instructions", lambda code: iter(instructions))
    monkeypatch.setattr("udfx.frontend.context.normalize", normalizer.normalize)
    assert trace(branch, {"s": STR}).value == apply("str", Col("s", STR))


@pytest.mark.parametrize("normalizer", [py313, py314], ids=["3.13", "3.14"])
def test_loop_cleanup_versions(monkeypatch: pytest.MonkeyPatch, normalizer: ModuleType) -> None:
    template = next(dis.get_instructions(branch))
    instructions = [
        template._replace(opname="END_FOR", offset=0),
        template._replace(opname="POP_ITER" if normalizer is py314 else "POP_TOP", offset=2),
    ]
    monkeypatch.setattr(dis, "get_instructions", lambda code: iter(instructions))
    assert [instruction.opname for instruction in normalizer.normalize(branch.__code__)] == [
        "END_FOR",
        "NOP",
    ]


def test_reduction_specialization_uses_call_path() -> None:
    template = next(dis.get_instructions(branch))
    instructions = [
        template._replace(opname="LOAD_GLOBAL", argval="any"),
        template._replace(opname="COPY", arg=1),
        template._replace(opname="LOAD_COMMON_CONSTANT", arg=4),
        template._replace(opname="IS_OP", arg=0),
        template._replace(opname="POP_JUMP_IF_FALSE", argval=100),
    ]
    normalized = py314.lower(instructions)
    assert [instruction.opname for instruction in normalized] == [
        "LOAD_GLOBAL",
        "NOP",
        "NOP",
        "NOP",
        "JUMP_FORWARD",
    ]
    assert normalized[-1].argval == 100
