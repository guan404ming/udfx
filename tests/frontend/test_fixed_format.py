# ruff: noqa: PLW0108, UP032

from collections.abc import Callable
from typing import Any

import pytest
from conftest import frontend_backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Call, Col, lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.testing.strategies import FLOAT_TRAPS
from udfx.types import FLOAT, STR


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: f"{value:.0f}",
        lambda value: f"{value:.2f}",
        lambda value: f"{value:.6f}",
        lambda value: f"{value:.22f}",
        lambda value: f"{value:.23f}",
        lambda value: "{:.2f}".format(value),
        lambda value: f"price: {value:.2f}",
    ],
)
def test_fixed_format_forms(function: Callable[..., object], backend: Backend[Any]) -> None:
    assert (
        check_function(
            function,
            {"value": FLOAT.or_none()},
            backend,
            [(value,) for value in (*FLOAT_TRAPS, None)],
        )
        == []
    )


def test_fixed_format_uses_shared_op() -> None:
    program = trace(lambda value: f"{value:.2f}", {"value": FLOAT})
    assert isinstance(program.value, Call)
    assert program.value.op.name == "format_fixed"
    assert program.result == STR


def test_literal_fixed_format_defers_allocation() -> None:
    program = trace(lambda: f"{1.2:.1000000000f}", {})
    assert isinstance(program.value, Call)
    assert program.value.op.name == "format_fixed"


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize("digits", [2, 6, 22])
def test_fractional_format_limits_input_expansion(backend: Backend[Any], digits: int) -> None:
    node = apply("format_fixed", Col("format_input", FLOAT), lit(digits))
    expression = backend.lower(node).expr
    limit = 1 if backend.name == "duckdb" else 7
    assert str(expression).count("format_input") <= limit
