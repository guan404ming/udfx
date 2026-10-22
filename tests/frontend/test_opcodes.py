from collections.abc import Callable, Iterator, Mapping
from types import CodeType

import pytest

from udfx.frontend import trace
from udfx.frontend.logic import fold
from udfx.frontend.machine import HANDLERS
from udfx.frontend.opcodes import normalize
from udfx.ir import FALSE, Col, If, Lit, Node
from udfx.ops import apply
from udfx.types import INT, STR, Type

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
