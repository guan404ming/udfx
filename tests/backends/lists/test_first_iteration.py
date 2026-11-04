import json
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR, list_of


def first(values: list[str]) -> int:
    for value in values:
        return int(value)
    return -1


def handled(values: list[str]) -> int:
    try:
        for value in values:
            return int(value)
    except ValueError:
        return -2
    return -1


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [first, handled])
def test_first_iteration_preserves_empty_and_fault_paths(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    rows = [(values,) for values in ([], ["1"], ["1", "invalid"], ["invalid", "1"])]
    assert check_function(function, {"values": list_of(STR)}, backend, rows) == []


def first_json_string(text: str) -> object:
    value = json.loads(text)
    if isinstance(value, list):
        for item in value:
            return item
    return None


@pytest.mark.parametrize("backend", backend_params())
def test_first_json_iteration_keeps_native_string_arrays(backend: Backend[Any]) -> None:
    schema = {"text": STR}
    program = trace(first_json_string, schema, result=STR.or_none())
    lowered = backend.lower_program(program)
    rows = [('["first","second"]',), ("[]",)]
    assert backend.evaluate([lowered.expr, backend.exact_expression(lowered)], schema, rows) == [
        ["first", None],
        [True, True],
    ]
