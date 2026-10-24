from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import INT64_MAX, INT64_MIN, STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda text: int(text) < 1,
        lambda text: int(text) <= 1,
        lambda text: int(text) > 1,
        lambda text: int(text) >= 1,
        lambda text: int(text) == 1,
        lambda text: int(text) != 1,
        lambda text: 1 < int(text),  # noqa: SIM300
        lambda text: 1 <= int(text),  # noqa: SIM300
        lambda text: 1 > int(text),  # noqa: SIM300
        lambda text: 1 >= int(text),  # noqa: SIM300
    ],
)
def test_parsed_comparison_avoids_intermediate_overflow(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    schema = {"text": STR.or_none()}
    program = trace(function, schema)
    assert backend.exact(backend.lower(program.value)) is None
    rows = [
        (text,)
        for text in (
            None,
            "0",
            "1",
            "2",
            "-1",
            str(INT64_MIN),
            str(INT64_MAX),
            str(2**200),
            str(-(2**200)),
            "\u0661_\u0662",
            "\u2003-\u0661_\u0662\u2003",
            "invalid",
            "9" * 5000,
        )
    ]
    assert check_function(function, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda text: int(text) == INT64_MAX,
        lambda text: int(text) > INT64_MAX,
        lambda text: int(text) == INT64_MIN,
        lambda text: int(text) < INT64_MIN,
    ],
)
def test_boundary_comparison_preserves_overflowing_values(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    rows = [(str(value),) for value in (INT64_MIN - 1, INT64_MIN, INT64_MAX, INT64_MAX + 1)]
    assert check_function(function, {"text": STR}, backend, rows) == []
