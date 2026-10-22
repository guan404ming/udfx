from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.ir import FALSE, Col, Comprehension, Element, Node
from udfx.ops import apply
from udfx.types import BOOL, STR, list_of


def nested(source: Node, depth: int) -> Comprehension:
    item = source.type.item
    assert item is not None
    element = Element(f"element{depth}", item)
    value = nested(element, depth - 1) if depth > 1 else apply(".strip", element)
    return Comprehension(source, element, value, Col("active", BOOL), FALSE)


@pytest.mark.parametrize("backend", backend_params())
def test_nested_comprehension_lowering_grows_linearly(backend: Backend[Any]) -> None:
    if backend.name == "polars":
        pytest.skip("Polars has no nested element scopes")
    sizes = []
    for depth in range(1, 5):
        type = STR
        row: object = " a "
        for _ in range(depth):
            type = list_of(type)
            row = [row]
        source = Col("values", type)
        node = nested(source, depth)
        with patch.object(backend, "lower", wraps=backend.lower) as lower:
            lowered = backend.lower(node)
        calls = [call.args[0] for call in lower.call_args_list]
        assert sum(isinstance(call, Comprehension) for call in calls) == depth
        assert len(calls) <= 7 * depth
        assert lowered.exact is None
        sizes.append(len(str(lowered.expr)))
        if backend.name == "datafusion" and depth > 1:
            continue
        expected: object = "a"
        for _ in range(depth):
            expected = [expected]
        assert backend.evaluate(
            [lowered.expr], {"values": type, "active": BOOL}, [(row, True)]
        ) == [[expected]]
    assert sizes[-1] <= 5 * sizes[0]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("guarded", [False, True])
def test_comprehension_body_lowers_once(backend: Backend[Any], guarded: bool) -> None:
    source = Col("words", list_of(STR))
    element = Element("word", STR)
    value = apply(".upper" if guarded else ".strip", element)
    condition = apply("bool", element)
    node = Comprehension(source, element, value, condition, FALSE)
    with patch.object(backend, "lower", wraps=backend.lower) as lower:
        lowered = backend.lower(node)
    assert sum(call.args[0] is value for call in lower.call_args_list) == 1
    assert sum(call.args[0] is condition for call in lower.call_args_list) == 1
    exact = backend.exact_expression(lowered)
    results, flags = backend.evaluate(
        [lowered.expr, exact], {"words": list_of(STR)}, [([" a ", ""],), ([],)]
    )
    assert results == [[" A " if guarded else "a"], []]
    assert flags == [True, True]
