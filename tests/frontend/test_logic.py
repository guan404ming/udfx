from collections.abc import Callable

import pytest

from udfx.frontend import trace
from udfx.frontend.calls import invoke
from udfx.frontend.logic import conjunction, disjunction, fold, negate
from udfx.ir import FALSE, TRUE, Col, Lit, lit
from udfx.ops import apply
from udfx.types import BOOL, INT, STR, list_of

X = Col("x", BOOL)
Y = Col("y", BOOL)
S = Col("s", STR.or_none())


def test_conj_drops_known_parts() -> None:
    both = conjunction(X, Y)
    assert conjunction(both, X) == both
    assert conjunction(X, TRUE) == X


def test_conj_finds_contradictions() -> None:
    assert conjunction(conjunction(X, Y), negate(X)) == FALSE
    assert conjunction(apply("bool", S), apply("is_none", S)) == FALSE


def test_truthy_implies_not_none() -> None:
    truthy = apply("bool", S)
    assert conjunction(truthy, negate(apply("is_none", S))) == truthy


def test_neg_removes_double_negation() -> None:
    assert negate(negate(X)) == X


def test_disj_drops_duplicates() -> None:
    assert disjunction(X, X) == X
    assert disjunction(FALSE, X) == X


def test_fold_evaluates_literal_calls() -> None:
    assert fold(apply("not", apply("bool", Lit("::", STR)))) == FALSE
    assert fold(apply("bool", S)) == apply("bool", S)


def test_fold_keeps_raising_calls() -> None:
    parse = apply("int", Lit("x", STR))
    assert fold(parse) == parse


@pytest.mark.parametrize("method", [".lower", ".upper"])
@pytest.mark.parametrize("strip", [".strip", ".lstrip", ".rstrip"])
def test_case_guard_precedes_strip(method: str, strip: str) -> None:
    assert fold(apply(method, apply(strip, S))) == apply(strip, apply(method, S))


@pytest.mark.parametrize("method", [".lower", ".upper"])
@pytest.mark.parametrize("strip", [".strip", ".lstrip", ".rstrip"])
def test_case_keeps_explicit_strip_characters(method: str, strip: str) -> None:
    node = apply(method, apply(strip, S, lit("A")))
    assert fold(node) == node


def test_constant_receiver_folds() -> None:
    t = Col("t", STR)
    program = trace(lambda t: "a,b ".strip().replace(",", t), {"t": STR})
    assert program.value == apply(".replace", Lit("a,b", STR), Lit(",", STR), t)


def test_constant_list_folds_inside_its_consumer() -> None:
    t = Col("t", STR)
    program = trace(lambda t: t in "a,b".upper().split(","), {"t": STR})
    assert program.value == apply("__contains__", Lit(("A", "B"), list_of(STR)), t)


def test_constant_call_folds() -> None:
    assert trace(lambda t: int("12"), {"t": STR}).value == Lit(12, INT)


def test_raising_constant_call_keeps_its_faults() -> None:
    node, faults = invoke("__getitem__", [lit("abc"), lit(5)], TypeError)
    assert node == apply("__getitem__", lit("abc"), lit(5))
    assert [fault.kind for fault in faults] == [IndexError]


@pytest.mark.parametrize(
    "function",
    [lambda text: text.rsplit(",")[0], lambda text: (text.split(",") + text.rsplit(","))[1]],
)
def test_minimum_length_removes_index_fault(function: Callable[[str], str]) -> None:
    assert trace(function, {"text": STR}).fault == FALSE
