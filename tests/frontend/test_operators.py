import pytest
from hypothesis import given

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import FALSE, Col, If, Lit
from udfx.ops import apply
from udfx.testing import batches, check_function
from udfx.types import BOOL, INT, STR, list_of

POLARS = PolarsBackend()

S = Col("s", STR.or_none())
A = Col("a", INT)
B = Col("b", INT)
ITEMS = {"item": STR, "items": list_of(STR).or_none()}


def upper_and(s: str | None) -> str | None:
    return s and s.upper()


def default_or(s: str | None) -> str:
    return s or "default"


def both_set(s: str | None, t: str | None) -> bool:
    return s is not None and t is not None


def negated(s: str | None) -> bool:
    return not s


def missing(s: str | None) -> bool:
    return s is None


def chained(a: int, b: int) -> bool:
    return a > b > 0


def negative(a: int) -> int:
    return -a


def has_colons(s: str) -> bool:
    return "::" in s


def lacks_colons(s: str) -> bool:
    return "::" not in s


def same(s: str | None, t: str | None) -> bool:
    return s is t


def member(item: str, items: list[str] | None) -> bool:
    return item in items  # pyright: ignore[reportOperatorIssue]


def nonempty(item: str, items: list[str] | None) -> bool:
    return bool(items)


def test_and_returns_operand() -> None:
    program = trace(upper_and, {"s": STR.or_none()})
    assert program.value == If(apply("bool", S), apply(".upper", S), S, STR.or_none())
    assert program.fault == FALSE


def test_or_returns_operand() -> None:
    program = trace(default_or, {"s": STR.or_none()})
    assert program.value == If(apply("bool", S), S, Lit("default", STR), STR.or_none())


def test_not_negates_truthiness() -> None:
    program = trace(negated, {"s": STR.or_none()})
    assert program.value == apply("not", apply("bool", S))


def test_is_none_uses_is_none_op() -> None:
    program = trace(missing, {"s": STR.or_none()})
    assert program.value == apply("is_none", S)


def test_is_none_of_non_nullable_is_false() -> None:
    program = trace(missing, {"s": STR})
    assert program.value == FALSE


def test_chained_comparison_uses_swap_and_copy() -> None:
    program = trace(chained, {"a": INT, "b": INT})
    first = apply("__gt__", A, B)
    second = apply("__gt__", B, Lit(0, INT))
    assert program.value == If(first, second, first, BOOL)


def test_unary_negative_uses_neg_op() -> None:
    assert trace(negative, {"a": INT}).value == apply("__neg__", A)


def test_contains_puts_container_first() -> None:
    s = Col("s", STR)
    found = apply("__contains__", s, Lit("::", STR))
    assert trace(has_colons, {"s": STR}).value == found
    assert trace(lacks_colons, {"s": STR}).value == apply("not", found)


def test_optional_list_tests_for_none() -> None:
    item, items = Col("item", STR), Col("items", ITEMS["items"])
    program = trace(member, ITEMS)
    assert program.value == apply("__contains__", items, item)
    assert program.fault == apply("is_none", items)
    assert trace(nonempty, ITEMS).value == apply("bool", items)


@given(batches(ITEMS))
def test_optional_list_matches_python(rows: list[tuple[object, ...]]) -> None:
    assert check_function(member, ITEMS, POLARS, rows) == []
    assert check_function(nonempty, ITEMS, POLARS, rows) == []


def test_identity_check_is_unsupported() -> None:
    with pytest.raises(Unsupported, match="identity"):
        trace(same, {"s": STR.or_none(), "t": STR.or_none()})


@given(batches({"s": STR.or_none()}))
def test_single_string_matches_python(rows: list[tuple[object, ...]]) -> None:
    for function in (upper_and, default_or, negated, missing):
        assert check_function(function, {"s": STR.or_none()}, POLARS, rows) == []


@given(batches({"s": STR.or_none(), "t": STR.or_none()}))
def test_both_set_matches_python(rows: list[tuple[object, ...]]) -> None:
    schema = {"s": STR.or_none(), "t": STR.or_none()}
    assert check_function(both_set, schema, POLARS, rows) == []


@given(batches({"a": INT, "b": INT}))
def test_chained_matches_python(rows: list[tuple[object, ...]]) -> None:
    assert check_function(chained, {"a": INT, "b": INT}, POLARS, rows) == []
