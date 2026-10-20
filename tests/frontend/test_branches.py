import pytest
from hypothesis import given

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.machine import MAX_STEPS
from udfx.ir import FALSE, Col, If, Lit
from udfx.ops import apply
from udfx.testing import batches, check_function
from udfx.types import INT, STR

POLARS = PolarsBackend()

S = Col("s", STR.or_none())
A = Col("a", INT)
B = Col("b", INT)


def lower_if_truthy(s: str | None) -> str | None:
    if s:
        return s.lower()
    return None


def upper_unless_none(s: str | None) -> str:
    if s is None:
        return "none"
    return s.upper()


def upper_if_not_none(s: str | None) -> str | None:
    if s is not None:
        return s.upper()
    return s


def larger(a: int, b: int) -> int:
    if a > b:
        return a
    else:
        return b


def upper_or_default(s: str | None) -> str:
    return (s if s else "x").upper()


def loop(a: int) -> int:
    while a > 0:
        a = a + 1
    return a


def test_truthiness_uses_bool_op() -> None:
    program = trace(lower_if_truthy, {"s": STR.or_none()})
    lowered = apply(".lower", Col("s", STR.or_none()))
    assert program.value == If(apply("bool", S), lowered, Lit(None, STR.or_none()), STR.or_none())
    assert program.fault == FALSE


def test_bool_condition_is_used_directly() -> None:
    program = trace(larger, {"a": INT, "b": INT})
    assert program.value == If(apply("__gt__", A, B), A, B, INT)


def test_pop_jump_if_not_none() -> None:
    program = trace(upper_unless_none, {"s": STR.or_none()})
    assert program.value == If(apply("is_none", S), Lit("none", STR), apply(".upper", S), STR)
    assert program.fault == FALSE


def test_pop_jump_if_none() -> None:
    program = trace(upper_if_not_none, {"s": STR.or_none()})
    assert program.value == If(apply("is_none", S), S, apply(".upper", S), STR.or_none())
    assert program.fault == FALSE


def test_jump_forward_joins_branches() -> None:
    program = trace(upper_or_default, {"s": STR.or_none()})
    assert program.value == If(apply("bool", S), apply(".upper", S), Lit("X", STR), STR)
    assert program.fault == FALSE


def test_loops_are_unsupported() -> None:
    with pytest.raises(Unsupported, match="loops"):
        trace(loop, {"a": INT})


@given(batches({"s": STR.or_none()}))
def test_truthiness_matches_python(rows: list[tuple[object, ...]]) -> None:
    assert check_function(lower_if_truthy, {"s": STR.or_none()}, POLARS, rows) == []
    assert check_function(upper_unless_none, {"s": STR.or_none()}, POLARS, rows) == []
    assert check_function(upper_if_not_none, {"s": STR.or_none()}, POLARS, rows) == []
    assert check_function(upper_or_default, {"s": STR.or_none()}, POLARS, rows) == []


@given(batches({"a": INT, "b": INT}))
def test_comparison_branch_matches_python(rows: list[tuple[object, ...]]) -> None:
    assert check_function(larger, {"a": INT, "b": INT}, POLARS, rows) == []


def many_branches(a: int) -> int:
    total = 0
    if a > 1:
        total += 1
    if a > 2:
        total += 2
    if a > 3:
        total += 3
    if a > 4:
        total += 4
    if a > 5:
        total += 5
    if a > 6:
        total += 6
    if a > 7:
        total += 7
    if a > 8:
        total += 8
    if a > 9:
        total += 9
    if a > 10:
        total += 10
    if a > 11:
        total += 11
    if a > 12:
        total += 12
    return total


def test_too_many_paths_names_the_cause() -> None:
    message = (
        f"more than {MAX_STEPS} execution paths; "
        "the function has too many independent branches for one expression"
    )
    with pytest.raises(Unsupported) as error:
        trace(many_branches, {"a": INT})
    assert str(error.value) == message
