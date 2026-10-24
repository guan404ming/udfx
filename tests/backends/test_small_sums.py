from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params
from corpus.cases_lists import total_numbers, unpack_numbers

from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import INT, INT64_MAX, INT64_MIN, STR, list_of


def sum_numbers(numbers: list[int]) -> int:
    return sum(numbers)


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [sum_numbers, unpack_numbers])
def test_small_sum_preserves_cancellation(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark scalar sums")
    rows = [
        (None,),
        ([],),
        ([0],),
        ([INT64_MIN],),
        ([INT64_MAX, INT64_MAX, INT64_MIN],),
        ([INT64_MIN, INT64_MIN, INT64_MAX, INT64_MAX],),
        ([INT64_MAX, INT64_MAX],),
        ([INT64_MIN, -1],),
        ([INT64_MAX, INT64_MAX, INT64_MIN, 0, 0],),
        ([1] * 128,),
    ]
    assert check_function(function, {"numbers": list_of(INT).or_none()}, backend, rows) == []


def test_small_sum_keeps_large_array_path() -> None:
    backend = SparkBackend()
    expression = str(backend.build(trace(sum_numbers, {"numbers": list_of(INT)})))
    assert "get(" in expression
    assert "aggregate(" in expression
    assert "patch(" not in expression


def test_small_sum_does_not_repeat_computed_array() -> None:
    expression = str(SparkBackend().build(trace(total_numbers, {"text": STR})))
    assert "get(" not in expression
