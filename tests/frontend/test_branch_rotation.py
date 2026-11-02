from itertools import product

import pytest

from udfx.backends.lowering_context import evaluated_nodes
from udfx.frontend.core.branches import factor_branches, shared_branch
from udfx.graphs import unique_size
from udfx.ir import FALSE, TRUE, Col, Guarded, If, children, lit
from udfx.ops import apply
from udfx.semantics.assumptions import CONDITIONS, assuming, known, predicates
from udfx.types import BOOL, STR


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("default", [False, True])
def test_nested_predicate_proves_the_selected_branch(positive: bool, default: bool) -> None:
    condition, selected = Col("condition", BOOL), Col("selected", BOOL)
    predicate = If(condition, selected, lit(default), BOOL)
    if not positive:
        predicate = If(condition, lit(default), selected, BOOL)
    with assuming(predicate, not default):
        found = dict(CONDITIONS.get())
        assert known(condition, found) is positive
        assert known(selected, found) is not default


def test_existing_facts_select_a_nested_predicate() -> None:
    condition, left, right = (Col(name, BOOL) for name in ("condition", "left", "right"))
    predicate = If(condition, left, right, BOOL)
    with assuming(left, False), assuming(predicate, True):
        found = dict(CONDITIONS.get())
        assert known(condition, found) is False
        assert known(right, found) is True
    with assuming(condition, True), assuming(predicate, False):
        assert known(left, dict(CONDITIONS.get())) is False
    with assuming(predicate, True):
        assert all(known(node, dict(CONDITIONS.get())) is None for node in (condition, left, right))


def test_selected_predicate_reuses_only_evaluated_guards() -> None:
    condition = Col("condition", BOOL)
    selected = Guarded(Col("value", BOOL), Col("valid", BOOL))
    predicate = If(condition, selected, FALSE, BOOL)
    with assuming(predicate, True):
        assert id(selected) in evaluated_nodes(predicate)
    with assuming(predicate, False):
        assert id(selected) not in evaluated_nodes(predicate)
    with assuming(condition, False):
        assert id(selected) not in evaluated_nodes(predicate)


def test_predicate_proofs_keep_shared_graphs_bounded() -> None:
    source = Col("text", STR)
    for _ in range(24):
        source = apply("__add__", source, source)
    selected = apply("bool", source)
    predicate = If(Col("condition", BOOL), selected, FALSE, BOOL)
    with assuming(predicate, True):
        assert len(evaluated_nodes(predicate)) <= 28
    shared = If(Col("other", BOOL), selected, FALSE, BOOL)
    node = If(Col("first", BOOL), If(Col("second", BOOL), TRUE, shared, BOOL), shared, BOOL)
    (factored,) = factor_branches((node,))
    assert unique_size((factored,)) <= unique_size((node,)) + 2
    pending = [(factored, 0)]
    seen = set()
    while pending:
        current, depth = pending.pop()
        assert depth <= 30
        if id(current) not in seen:
            seen.add(id(current))
            pending.extend((child, depth + 1) for child in children(current))


def test_nested_predicates_collect_shared_facts_once() -> None:
    node = Col("value", BOOL)
    for _ in range(24):
        node = If(node, node, FALSE, BOOL)
    assert len(predicates(node, True)) == 25
    with assuming(node, True):
        assert known(Col("value", BOOL), dict(CONDITIONS.get())) is True


@pytest.mark.parametrize("nullable", [False, True])
def test_literal_default_and_nullable_conditions_keep_their_shape(nullable: bool) -> None:
    condition = Col("condition", BOOL.or_none() if nullable else BOOL)
    shared = Col("shared", BOOL) if nullable else FALSE
    node = If(condition, If(Col("second", BOOL), TRUE, shared, BOOL), shared, BOOL)
    assert shared_branch(node) is node


def test_nested_predicate_implications_hold_for_all_assignments() -> None:
    first, second, third = (Col(name, BOOL) for name in ("first", "second", "third"))
    predicate = If(first, If(second, third, FALSE, BOOL), second, BOOL)
    for first_value, second_value, third_value in product((False, True), repeat=3):
        result = (third_value if second_value else False) if first_value else second_value
        with assuming(first, first_value), assuming(predicate, result):
            found = dict(CONDITIONS.get())
            for node, value in ((second, second_value), (third, third_value)):
                inferred = known(node, found)
                assert inferred is None or inferred == value
