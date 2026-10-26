from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from bench.inspection import fallback_in_plan
from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.testing import check_function
from udfx.types import INT, INT64_MAX, INT64_MIN, list_of


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda values: [value * value for value in values if -100 <= value <= 100],
        lambda values: [value + 1 for value in values if value < 1000],
    ],
)
def test_filtered_integer_bounds_remove_fallback(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    schema = {"values": list_of(INT)}
    translation = compile(function, schema, backend)
    assert translation.ok, translation.unsupported
    if isinstance(backend, SparkBackend):
        assert not fallback_in_plan(translation.expr, "spark")
    rows = [([],), ([INT64_MIN, -1000, -100, -1, 0, 1, 100, 1000, INT64_MAX],)]
    assert check_function(function, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_comprehension_bounds_stay_in_filtered_value(backend: Backend[Any]) -> None:
    def compared(values: list[int]) -> list[bool]:
        return [value + 1 > 0 for value in values if value < 1000] + [
            value + 1 > 0 for value in values
        ]

    schema = {"values": list_of(INT)}
    rows = [([],), ([INT64_MAX],), ([INT64_MIN, -1, 0, 1000, INT64_MAX],)]
    assert check_function(compared, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_comprehension_bounds_preserve_filter_faults(backend: Backend[Any]) -> None:
    def divided(values: list[int]) -> list[int]:
        return [value + 1 for value in values if value < 1000 and 10 // value]

    schema = {"values": list_of(INT)}
    rows = [([],), ([0],), ([INT64_MAX, -1, 0],), ([INT64_MIN, 1, 10, 1000],)]
    assert check_function(divided, schema, backend, rows) == []
