from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx import compile
from udfx.backends.base import Backend
from udfx.types import INT, STR, Type


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("function", "schema", "size"),
    [
        (lambda s: s // 60 // 60 // 24 // 7, {"s": INT}, 6679),
        (lambda t: t[1:][1:][1:][1:], {"t": STR}, 275608),
        (lambda a, b: min(max(min(max(a, b), a), b), a), {"a": INT, "b": INT}, 116),
    ],
    ids=["division", "slice", "extrema"],
)
def test_expression_size(
    backend: Backend[Any], function: Callable[..., object], schema: dict[str, Type], size: int
) -> None:
    translation = compile(function, schema, backend)
    assert translation.ok
    assert len(str(translation.expr)) < 2 * size
