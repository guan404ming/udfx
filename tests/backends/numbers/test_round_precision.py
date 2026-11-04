from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.ir import Col, Node, lit
from udfx.ops import apply
from udfx.types import FLOAT, INT

BACKENDS = [value for value in backend_params() if value.id in {"duckdb", "datafusion"}]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("digits", [1, 4, 22])
def test_positive_precision_uses_native_round(backend: Backend[Any], digits: int) -> None:
    lowered = backend.lower(apply("round", Col("value", FLOAT), lit(digits)))
    assert str(lowered.expr).lower().count("round(") == (2 if backend.name == "datafusion" else 1)
    assert lowered.exact is not None


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("digits", [lit(0), lit(-1), Col("digits", INT)])
def test_zero_negative_dynamic_precision_keeps_half_even(
    backend: Backend[Any], digits: Node
) -> None:
    lowered = backend.lower(apply("round", Col("value", FLOAT), digits))
    assert str(lowered.expr).lower().count("round(") == (4 if backend.name == "datafusion" else 2)
    assert lowered.exact is not None
