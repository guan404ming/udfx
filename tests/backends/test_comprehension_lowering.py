from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.hoisting import invariant
from udfx.ir import FALSE, TRUE, Col, Comprehension, Element, Node
from udfx.ops import apply
from udfx.types import BOOL, INT, STR, list_of


def nested(source: Node, depth: int) -> Comprehension:
    item = source.type.item
    assert item is not None
    element = Element(f"element{depth}", item)
    value = nested(element, depth - 1) if depth > 1 else apply(".strip", element)
    return Comprehension(source, element, value, Col("active", BOOL), FALSE)


@pytest.mark.parametrize("backend", backend_params())
def test_nested_comprehension_lowering_evaluates(backend: Backend[Any]) -> None:
    if backend.name == "polars":
        pytest.skip("Polars has no nested element scopes")
    for depth in range(1, 5):
        type = STR
        row: object = " a "
        for _ in range(depth):
            type = list_of(type)
            row = [row]
        source = Col("values", type)
        node = nested(source, depth)
        lowered = backend.lower(node)
        assert lowered.exact is None
        if backend.name == "datafusion" and depth > 1:
            continue
        expected: object = "a"
        for _ in range(depth):
            expected = [expected]
        assert backend.evaluate(
            [lowered.expr], {"values": type, "active": BOOL}, [(row, True)]
        ) == [[expected]]


def test_invariant_skips_inner_elements_and_picks_outer_calls() -> None:
    words = apply(".split", Col("text", STR))
    element = Element("index", INT)
    value = apply("__getitem__", words, element)
    node = Comprehension(Col("indexes", list_of(INT)), element, value, TRUE, FALSE)
    assert invariant(node, {}) == words
    assert invariant(node, {words: None}) is None
    assert invariant(nested(Col("values", list_of(list_of(STR))), 2), {}) is None
