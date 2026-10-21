import dis
from collections.abc import Callable, Iterator, Mapping
from types import CodeType, ModuleType

import pytest

from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.logic import fold
from udfx.frontend.machine import HANDLERS
from udfx.frontend.opcodes import normalize, py312, py313, py314
from udfx.ir import FALSE, Col, If, Lit, Node
from udfx.ops import apply
from udfx.types import FLOAT, INT, STR, Type

S = Col("s", STR.or_none())
T = Col("t", STR)
A = Col("a", INT)
B = Col("b", INT)
NUMBER = Col("number", INT)
MISSING = apply("is_none", S)
ZERO = Lit(0, INT)
(OUT_OF_RANGE,) = apply("__getitem__", T, ZERO).op.faults(T, ZERO)


def branch(s: str | None) -> str | None:
    if s:
        return s.lower()
    return None


def helper(t: str) -> str:
    def up(word: str) -> str:
        return word.upper()

    return up(t)


def closure(a: int) -> bool:
    limit = 0

    def big(value: int) -> bool:
        return value > limit

    return big(a)


def guarded(s: str) -> str:
    try:
        return s.lower()
    except AttributeError:
        return "none"


def middle(t: str) -> str:
    return t[1:3]


def first(t: str) -> str:
    return t[0]


def has_colons(t: str) -> bool:
    return "::" in t


def unpack(a: int, b: int) -> int:
    x, y = a, b
    return x + y


def truth(s: str | None) -> bool:
    return not not s  # noqa: SIM208


def formatted(number: int) -> str:
    return f"<{number:d}>"


def converted(number: int) -> str:
    return f"{number!r:s}"


def keyword_round(number: float) -> float:
    return round(number, ndigits=2)


CASES: list[tuple[Callable[..., object], Mapping[str, Type], Node, Node]] = [
    (
        branch,
        {"s": S.type},
        If(apply("bool", S), apply(".lower", S), Lit(None, S.type), S.type),
        FALSE,
    ),
    (helper, {"t": STR}, apply(".upper", T), FALSE),
    (closure, {"a": INT}, apply("__gt__", A, ZERO), FALSE),
    (guarded, {"s": S.type}, If(MISSING, Lit("none", STR), apply(".lower", S), STR), FALSE),
    (middle, {"t": STR}, apply("slice", T, Lit(1, INT), Lit(3, INT)), FALSE),
    (first, {"t": STR}, apply("__getitem__", T, ZERO), fold(OUT_OF_RANGE.when)),
    (has_colons, {"t": STR}, apply("__contains__", T, Lit("::", STR)), FALSE),
    (unpack, {"a": INT, "b": INT}, apply("__add__", A, B), FALSE),
    (truth, {"s": S.type}, apply("bool", S), FALSE),
    (
        formatted,
        {"number": INT},
        apply("__add__", apply("__add__", Lit("<", STR), apply("str", NUMBER)), Lit(">", STR)),
        FALSE,
    ),
    (converted, {"number": INT}, apply("str", NUMBER), FALSE),
]


def codes(code: CodeType) -> Iterator[CodeType]:
    yield code
    for constant in code.co_consts:
        if isinstance(constant, CodeType) and constant.co_name != "__annotate__":
            yield from codes(constant)


@pytest.mark.parametrize("function", [case[0] for case in CASES])
def test_normalize_emits_canonical_opcodes(function: Callable[..., object]) -> None:
    for code in codes(function.__code__):
        assert {instruction.opname for instruction in normalize(code)} <= set(HANDLERS)


@pytest.mark.parametrize(("function", "schema", "value", "fault"), CASES)
def test_trace_matches_expected_ir(
    function: Callable[..., object], schema: Mapping[str, Type], value: Node, fault: Node
) -> None:
    program = trace(function, schema)
    assert program.value == value
    assert program.fault == fault


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
