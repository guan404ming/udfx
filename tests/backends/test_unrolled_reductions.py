from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import INT, INT64_MAX, INT64_MIN, STR, list_of


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda items: sum(value * weight for value, weight in zip(items, (1, 2), strict=False)),
        lambda items: sum(value for value, _index in zip(items, (0, 1), strict=False) if value > 0),
    ],
)
def test_unrolled_sum_preserves_overflow(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    schema = {"items": list_of(INT).or_none()}
    rows = [
        ([],),
        ([1],),
        ([1, 2, 3],),
        ([INT64_MAX, INT64_MIN],),
        ([INT64_MIN, INT64_MAX],),
        ([INT64_MAX, INT64_MAX],),
        (None,),
    ]
    assert check_function(function, schema, backend, rows) == []


def test_spark_unrolled_sum_skips_array_aggregation() -> None:
    def weighted(text: str) -> int:
        return sum(len(word) * weight for word, weight in zip(text.split(), (1, 2), strict=False))

    backend = SparkBackend()
    schema = {"text": STR}
    expression = str(backend.build(trace(weighted, schema)))
    assert "aggregate(" not in expression
    assert expression.count("transform(") == 1
    assert expression.count("regexp_extract_all(") == 2
    assert "patch(" not in expression
    assert check_function(weighted, schema, backend, [("",), ("one",), ("a bb ccc",)]) == []
