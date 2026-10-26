from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from conftest import backend_params

from bench.inspection import fallback_in_plan
from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.frontend import trace
from udfx.ir import FALSE, Guarded, lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import INT, INT64_MAX, INT64_MIN, STR, list_of
from workloads.corpus.cases_lists import unpack_numbers
from workloads.corpus.cases_strings import index_sliced_parts


def negative_index(index: int, text: str) -> int:
    return index + len(text) if index < 0 else index


def positive_offset(index: int, text: str) -> int:
    return index - len(text) if index >= 0 else index


def repeated_expression(index: int, text: str) -> int:
    value = len(text)
    if index < 0:
        return index + value
    return (index + value) // 2


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [negative_index, positive_offset])
def test_branch_bounds_remove_safe_overflow(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    schema = {"index": INT, "text": STR}
    translation = compile(function, schema, backend)
    assert translation.ok, translation.unsupported
    if isinstance(backend, SparkBackend):
        assert not fallback_in_plan(translation.expr, "spark")
    rows = [
        (index, text) for index in (INT64_MIN, -1, 0, 1, INT64_MAX) for text in ("", "abc", "😀")
    ]
    assert check_function(function, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_lowering_cache_keeps_branch_bounds_separate(backend: Backend[Any]) -> None:
    schema = {"index": INT, "text": STR}
    translation = compile(repeated_expression, schema, backend)
    assert translation.ok, translation.unsupported
    if isinstance(backend, SparkBackend):
        assert fallback_in_plan(translation.expr, "spark")
    rows = [(index, "abc") for index in (INT64_MIN, -1, 0, 1, INT64_MAX)]
    assert check_function(repeated_expression, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_range_comparison_keeps_interior_equality(backend: Backend[Any]) -> None:
    def interior(index: int, text: str) -> bool:
        value = index if -1 <= index <= 1 else len(text) % 2
        return value == 0

    schema = {"index": INT, "text": STR}
    rows = [(index, text) for index in (-2, -1, 0, 1, 2) for text in ("", "abc")]
    assert check_function(interior, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_search_bounds_keep_offsets_native(backend: Backend[Any]) -> None:
    def search(text: str) -> int:
        return (text.find("a") + text.rfind("b") + text.count("")) // 2

    schema = {"text": STR}
    translation = compile(search, schema, backend)
    assert translation.ok, translation.unsupported
    if isinstance(backend, SparkBackend):
        assert not fallback_in_plan(translation.expr, "spark")
    assert check_function(search, schema, backend, [(text,) for text in ("", "ab", "😀ab")]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_guarded_partition_uses_fault_conditions(backend: Backend[Any]) -> None:
    schema = {"numbers": list_of(INT).or_none()}
    translation = compile(unpack_numbers, schema, backend)
    assert translation.ok, translation.unsupported
    if isinstance(backend, SparkBackend):
        assert not fallback_in_plan(translation.expr, "spark")
    rows = [(value,) for value in (None, [], [1], [1, 2], [INT64_MAX, 1, -1], [INT64_MAX, 1])]
    assert check_function(unpack_numbers, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_slice_index_bounds_keep_offsets_native(backend: Backend[Any]) -> None:
    schema = {"text": STR, "index": INT}
    translation = compile(index_sliced_parts, schema, backend)
    assert translation.ok, translation.unsupported
    if isinstance(backend, SparkBackend):
        assert not fallback_in_plan(translation.expr, "spark")
    rows = [(text, index) for text in ("", "a_b_c_d") for index in (INT64_MIN, -2, 0, 2, INT64_MAX)]
    assert check_function(index_sliced_parts, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_range_comparison_keeps_approximation_guard(backend: Backend[Any]) -> None:
    schema = {"value": INT}
    program = trace(lambda value: value == 1, schema)
    approximation = apply("__eq__", Guarded(lit(0), FALSE), lit(1))
    expression = backend.build(replace(program, value=approximation))
    assert backend.evaluate([expression], schema, [(0,), (1,), (2,)]) == [[False, True, False]]
