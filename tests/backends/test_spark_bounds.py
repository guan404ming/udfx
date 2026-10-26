from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from bench.inspection import fallback_in_plan
from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import INT, INT64_MAX, INT64_MIN, STR, list_of
from workloads.corpus.cases_lists import total_word_length, weighted_word_length


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function", [lambda text: (len(text) + 2) * 3, total_word_length, weighted_word_length]
)
def test_spark_bounded_arithmetic_stays_native(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark integer bounds")
    schema = {"text": STR.or_none()}
    translation = compile(function, schema, backend)
    assert translation.ok, translation.unsupported
    assert not fallback_in_plan(translation.expr, "spark")
    assert "decimal(38,0)" not in str(translation.expr)
    assert check_function(function, schema, backend, [(None,), ("",), ("a b",), ("😀 ü",)]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_spark_length_arithmetic_keeps_intermediate_overflow(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark integer bounds")

    def length(text: str) -> int:
        return (len(text) * INT64_MAX) // INT64_MAX

    schema = {"text": STR}
    translation = compile(length, schema, backend)
    assert translation.ok, translation.unsupported
    assert fallback_in_plan(translation.expr, "spark")
    assert check_function(length, schema, backend, [("",), ("a",), ("ab",), ("abc",)]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_spark_indexed_transform_skips_pairs(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark indexed transform")

    def indices(items: list[str]) -> list[int]:
        return [index for index, _ in enumerate(items)]

    schema = {"items": list_of(STR)}
    expression = str(backend.lower(trace(indices, schema).value).expr)
    assert "struct(" not in expression
    assert expression.count("transform(") == 1
    assert check_function(indices, schema, backend, [([],), (["a", "b"],)]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_bounded_padding_keeps_native_plan(backend: Backend[Any]) -> None:
    def padded(text: str, width: int) -> str:
        return text.zfill(width % 17).center(width % 31, "*")

    schema = {"text": STR, "width": INT}
    translation = compile(padded, schema, backend)
    assert translation.ok, translation.unsupported
    assert not fallback_in_plan(translation.expr, backend.name)
    rows = [
        (text, width) for text in ("", "-1", "😀") for width in (INT64_MIN, -1, 0, 30, INT64_MAX)
    ]
    assert check_function(padded, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("container", [STR, list_of(STR)])
def test_spark_index_fault_keeps_native_plan(backend: Backend[Any], container: Any) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark index bounds")

    def indexed(value: Any, index: int) -> str | None:
        try:
            return value[index]
        except IndexError:
            return None

    schema = {"value": container, "index": INT}
    translation = compile(indexed, schema, backend)
    assert translation.ok, translation.unsupported
    assert not fallback_in_plan(translation.expr, "spark")
    values = ("", "a😀") if container == STR else ([], ["a", "😀"])
    rows = [
        (value, index) for value in values for index in (INT64_MIN, -3, -2, -1, 0, 1, 2, INT64_MAX)
    ]
    assert check_function(indexed, schema, backend, rows) == []
