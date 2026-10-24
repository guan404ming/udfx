from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR, list_of


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda left, right: len(set(left) | set(right)) == 0,
        lambda left, right: 0 != len(set(left) | set(right)),  # noqa: SIM300
        lambda left, right: len(set(left) & set(right)) / len(set(left) | set(right)),
    ],
)
def test_empty_union_preserves_division_faults(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    schema = {"left": list_of(STR).or_none(), "right": list_of(STR).or_none()}
    rows = [
        ([], []),
        ([], ["a"]),
        (["a"], []),
        (["a", "a"], ["a", "b"]),
        (["ß", "ς"], ["\u03c3"]),
        (None, []),
        ([], None),
        (None, None),
    ]
    assert check_function(function, schema, backend, rows) == []


def test_empty_union_keeps_backend_choices() -> None:
    def empty(left: list[str], right: list[str]) -> bool:
        return len(set(left) | set(right)) == 0

    program = trace(empty, {"left": list_of(STR), "right": list_of(STR)})
    assert "set_union_size" in repr(program.value)
