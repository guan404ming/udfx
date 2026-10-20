from udfx.frontend.logic import conjunction, disjunction, fold, negate
from udfx.ir import FALSE, TRUE, Col, Lit
from udfx.ops import apply
from udfx.types import BOOL, STR

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
