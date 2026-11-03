from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from bench.inspection import fallback_in_plan
from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.ir import Col, Lit, lit
from udfx.ops import apply
from udfx.semantics.facts import integer_operation_fits
from udfx.testing import check_function
from udfx.types import INT, INT64_MAX, INT64_MIN, STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda left, right: (left + right) * 2 - left,
        lambda left, right: (left + right) - right,
    ],
)
def test_integer_regions_preserve_wide_intermediates(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    schema = {"left": INT, "right": INT}
    translation = compile(function, schema, backend)
    assert translation.ok, translation.unsupported
    if isinstance(backend, SparkBackend):
        assert not fallback_in_plan(translation.expr, "spark")
    values = (INT64_MIN, INT64_MIN + 1, -1, 0, 1, INT64_MAX - 1, INT64_MAX)
    assert (
        check_function(
            function, schema, backend, [(left, right) for left in values for right in values]
        )
        == []
    )


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda left, right: (left * right) * left,
        lambda left, right: (left + right) * 2 > left,
    ],
)
def test_integer_regions_keep_boundary_fallback(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    schema = {"left": INT, "right": INT}
    translation = compile(function, schema, backend)
    assert translation.ok, translation.unsupported
    if isinstance(backend, SparkBackend):
        assert fallback_in_plan(translation.expr, "spark")
    values = (INT64_MIN, -1, 0, 1, INT64_MAX)
    assert (
        check_function(
            function, schema, backend, [(left, right) for left in values for right in values]
        )
        == []
    )


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: (value // 100) * 60 + value % 100,
        lambda value: (value // 256) * 128 - value % 128,
    ],
)
def test_bounded_integer_arithmetic_avoids_fallback(
    backend: Backend[Any], function: Callable[[int], int]
) -> None:
    schema = {"value": INT}
    translation = compile(function, schema, backend)
    assert translation.ok, translation.unsupported
    if backend.name in {"duckdb", "spark"}:
        assert not fallback_in_plan(translation.expr, backend.name)
    values = (INT64_MIN, INT64_MIN + 1, -257, -101, -1, 0, 101, 257, INT64_MAX)
    assert check_function(function, schema, backend, [(value,) for value in values]) == []


def test_missing_slice_bound_keeps_integer_operation_fit() -> None:
    text = Col("text", STR)
    sliced = apply("slice", text, apply(".find", text, lit("a")), Lit(None, INT.or_none()))
    assert integer_operation_fits(
        apply("__sub__", apply("len", sliced), lit(5)), size_limit=INT64_MAX
    )
