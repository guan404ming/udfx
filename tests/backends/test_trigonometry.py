import math
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.semantics.math import SMALL_ANGLE
from udfx.testing import check_function
from udfx.types import FLOAT


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("function", "columns", "rows", "expected"),
    [
        (
            math.sin,
            ["value"],
            [(0.0,), (-0.0,), (SMALL_ANGLE,), (-0.6030047356199073,)],
            [True, True, True, False],
        ),
        (
            math.cos,
            ["value"],
            [(0.0,), (-SMALL_ANGLE,), (-0.5632819055357485,)],
            [True, True, False],
        ),
        (
            math.acos,
            ["value"],
            [(-1.0,), (0.0,), (1.0,), (-0.5363952858520995,)],
            [True, True, True, False],
        ),
        (
            math.atan2,
            ["vertical", "horizontal"],
            [
                (1.0, 1.0),
                (1.0, 0.0),
                (-0.0, 1.0),
                (-0.0, -1.0),
                (0.0, -0.0),
                (-0.2702105552426477, 91.02256930852431),
            ],
            [True, True, True, False, False, False],
        ),
    ],
)
def test_trigonometric_precision_domains(
    backend: Backend[Any],
    function: Callable[..., float],
    columns: list[str],
    rows: list[tuple[float, ...]],
    expected: list[bool],
) -> None:
    def unary(value: float) -> float:
        return function(value)

    def binary(vertical: float, horizontal: float) -> float:
        return function(vertical, horizontal)

    schema = dict.fromkeys(columns, FLOAT)
    program = trace(unary if len(columns) == 1 else binary, schema)
    lowered = backend.lower(program.value)
    if backend.name == "spark":
        assert lowered.exact is not None
        assert backend.evaluate([backend.predicate(lowered.exact)], schema, rows) == [expected]
        assert check_function(unary if len(columns) == 1 else binary, schema, backend, rows) == []
    else:
        assert lowered.exact is None
