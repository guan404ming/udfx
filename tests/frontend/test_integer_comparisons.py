import math
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from bench.inspection import fallback_in_plan
from udfx import compile
from udfx.backends.base import Backend
from udfx.testing import check_function
from udfx.types import INT, INT64_MAX, INT64_MIN, STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: value < 1.5,
        lambda value: 1.5 <= value,  # noqa: SIM300
        lambda value: value > -1.5,
        lambda value: -1.5 >= value,  # noqa: SIM300
        lambda value: value == 1.5,
        lambda value: value != 1.5,
        lambda value: value == 1.0,
        lambda value: value != -1.0,
        lambda value: value == math.inf,
        lambda value: value < math.inf,
        lambda value: value > -math.inf,
        lambda value: value >= math.nan,  # noqa: PLW0177
        lambda value: value < 2.0**63,
        lambda value: value == 2.0**53,
        lambda value: value != math.nan,  # noqa: PLW0177
    ],
)
def test_integer_comparisons_use_native_integer_rules(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    schema = {"value": INT.or_none()}
    translation = compile(function, {"value": INT}, backend)
    assert translation.ok, translation.unsupported
    assert not fallback_in_plan(translation.expr, backend.name)
    rows = [(value,) for value in (None, INT64_MIN, -2, -1, 0, 1, 2, INT64_MAX)]
    assert check_function(function, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_constant_comparison_keeps_operand_faults(backend: Backend[Any]) -> None:
    def different(text: str) -> bool:
        return int(text) != 1.5

    schema = {"text": STR}
    rows = [(text,) for text in ("1", "0", str(2**100), "invalid", "9" * 5000)]
    assert check_function(different, schema, backend, rows) == []
