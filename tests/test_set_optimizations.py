from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx import compile
from udfx.backends.base import Backend
from udfx.explain import render
from udfx.testing import check_function
from udfx.types import STR, Type, list_of


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda left, right, item: item in (set(left) | set(right)),
        lambda left, right, item: item in (set(left) & set(right)),
        lambda left, right, item: item in (set(left) - set(right)),
        lambda left, right, item: item in (set(left) ^ set(right)),
        lambda left, right, item: item in ((set(left) | set(right)) - set(left)),
    ],
)
def test_set_membership_skips_materialization(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    schema = {"left": list_of(STR).or_none(), "right": list_of(STR).or_none(), "item": STR}
    translation = compile(function, schema, backend)
    assert translation.program is not None, translation.unsupported
    assert "set_" not in render(translation.program.value)
    rows = [
        (left, right, item)
        for left in (None, [], ["a", "a", "b"])
        for right in (None, [], ["a", "c"])
        for item in ("a", "b", "c", "missing")
    ]
    assert check_function(function, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda left, right, last: sorted((set(left) | set(right)) - set(last)),
        lambda left, right, last: len(set(left).union(right).union(last).difference(right)),
        lambda left, right, last: (set(left) | set(right)).issubset(set(last)),
    ],
)
def test_set_consumers_skip_intermediate_union(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    schema: dict[str, Type] = dict.fromkeys(("left", "right", "last"), list_of(STR).or_none())
    translation = compile(function, schema, backend)
    assert translation.program is not None, translation.unsupported
    assert "set_union" not in render(translation.program.value)
    rows = [
        (left, right, last)
        for left in (None, [], ["b", "a", "a"])
        for right in ([], ["c", "a", "c"])
        for last in ([], ["a", "b", "c"], ["b"])
    ]
    assert check_function(function, schema, backend, rows) == []
