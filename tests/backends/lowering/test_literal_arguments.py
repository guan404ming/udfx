from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import FLOAT, STR


@pytest.mark.parametrize("backend", backend_params())
def test_literal_index_compares_positions(backend: Backend[Any]) -> None:
    def position(key: str) -> int:
        modes = ["AIR", "MAIL", "AIR", ""]
        return modes.index(key) if key in modes else -1

    schema = {"key": STR}
    expression = str(backend.lower_program(trace(position, schema)).expr).lower()
    for search in ("list_position", "array_position"):
        assert search not in expression
    rows = [("AIR",), ("MAIL",), ("",), ("air",), ("é",)]
    assert check_function(position, schema, backend, rows) == []


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"datafusion", "duckdb"}]
)
def test_integral_literal_power_skips_complex_guard(backend: Backend[Any]) -> None:
    def squared(value: float) -> float:
        return value**2.0

    schema = {"value": FLOAT}
    assert "floor" not in str(backend.lower_program(trace(squared, schema)).expr).lower()
    rows = [(-3.5,), (0.0,), (-0.0,), (1e200,), (float("nan"),), (float("-inf"),)]
    assert check_function(squared, schema, backend, rows) == []
