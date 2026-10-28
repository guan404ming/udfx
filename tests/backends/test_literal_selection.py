from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Col, Guarded, If, lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import BOOL, FLOAT, INT, STR, Type


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "datafusion"}]
)
@pytest.mark.parametrize(
    ("kind", "keys", "other"),
    [(STR, ("red", "green"), "other"), (INT, (-1, 2), 0)],
)
def test_literal_selection_handles_nullable_source(
    backend: Backend[Any], kind: Type, keys: tuple[Any, Any], other: object
) -> None:
    choices = {keys[0]: -0.0, keys[1]: 1.5}

    def lookup(value: Any) -> float:
        return choices.get(value, 8.5)

    schema = {"value": kind.or_none()}
    rows = [(keys[0],), (keys[1],), (other,), (None,)]
    with patch.object(backend, "select_value", wraps=backend.select_value) as selected:
        assert check_function(lookup, schema, backend, rows) == []
    selected.assert_called_once()
    assert backend.lower_program(trace(lookup, schema)).exact is None


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "datafusion"}]
)
def test_literal_selection_preserves_source_guards(backend: Backend[Any]) -> None:
    value = Col("value", INT)
    valid = Col("valid", BOOL)
    source = Guarded(apply("__add__", value, lit(1)), valid)
    node = If(
        apply("__eq__", source, lit(1)),
        lit(1.5),
        If(apply("__eq__", source, lit(2)), lit(2.5), lit(0.0), FLOAT),
        FLOAT,
    )
    with patch.object(backend, "select_value", wraps=backend.select_value) as selected:
        lowered = backend.lower(node)
    selected.assert_called_once()
    assert lowered.exact is not None
    assert lowered.overflow is None
    schema = {"value": INT, "valid": BOOL}
    rows = [(0, True), (1, True), (3, True), (1, False), (2**63 - 1, True)]
    assert backend.evaluate([lowered.exact], schema, rows) == [[True, True, True, False, False]]
    assert backend.evaluate([lowered.expr], schema, rows[:3]) == [[1.5, 2.5, 0.0]]


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "datafusion"}]
)
@pytest.mark.parametrize("nullable", [False, True])
def test_literal_selection_keeps_null_results(backend: Backend[Any], nullable: bool) -> None:
    choices = {"first": None if nullable else "one", "second": "two"}

    def lookup(value: Any) -> str | None:
        return choices.get(value)

    rows = [("first",), ("second",), ("other",), (None,)]
    assert check_function(lookup, {"value": STR.or_none()}, backend, rows) == []


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "datafusion"}]
)
def test_literal_selection_keeps_float_comparison_semantics(backend: Backend[Any]) -> None:
    def lookup(value: float) -> str:
        if value == 1:
            return "one"
        if value == 2.0:
            return "two"
        return "other"

    rows = [(1.0,), (2.0,), (3.0,), (float("nan"),), (float("inf"),)]
    with patch.object(backend, "select_value", wraps=backend.select_value) as selected:
        assert check_function(lookup, {"value": FLOAT}, backend, rows) == []
    selected.assert_not_called()
