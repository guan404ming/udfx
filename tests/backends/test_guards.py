from collections.abc import Callable
from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import FALSE, TRUE, Col, Guarded, If, lit
from udfx.ops import apply
from udfx.program import Program
from udfx.types import BOOL, DATETIME, FLOAT, INT, INT64_MAX, STR, Type


def inverse(value: float) -> float | None:
    try:
        return 1 / value
    except ZeroDivisionError:
        return None


def constant_part(value: int) -> str | None:
    if value == 0:
        return "A"
    if value == 1:
        return "B"
    return None


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("function", "type", "rows"),
    [
        (lambda value: value == 0, FLOAT, [(0.0,), (-0.0,), (1.0,), (float("nan"),)]),
        (inverse, FLOAT, [(0.0,), (2.0,), (-1.0,)]),
        (constant_part, INT, [(0,), (1,), (INT64_MAX,)]),
        (lambda value: min(max(value, 0.0), 1.0), FLOAT, [(-0.0,), (0.0,), (2.0,), (0.5,)]),
        (lambda value: value.strftime("%Y-%m-%d"), DATETIME, [(datetime(2024, 2, 29),)]),
    ],
    ids=["is_zero", "inverse", "constant_part", "clamp", "strftime"],
)
def test_static_guards_build_without_fallback(
    backend: Backend[Any], function: Callable[..., object], type: Type, rows: list[tuple[Any, ...]]
) -> None:
    program = trace(function, {"value": type})
    with patch.object(backend, "fallback", wraps=backend.fallback) as fallback:
        expr = backend.build(program)
    fallback.assert_not_called()
    assert backend.evaluate([expr], program.schema, rows) == [program.run(rows)]


@pytest.mark.parametrize("backend", backend_params())
def test_boolean_guards_fold_before_lowering(backend: Backend[Any]) -> None:
    value = Col("value", STR)
    guard = apply("not", apply("not", apply("or", TRUE, apply("is_none", value))))
    fault = apply("and", FALSE, apply("is_none", value))
    program = Program(lambda value: value, {"value": STR}, Guarded(value, guard), fault)
    with patch.object(backend, "fallback", wraps=backend.fallback) as fallback:
        expr = backend.build(program)
    fallback.assert_not_called()
    assert backend.evaluate([expr], program.schema, [("text",)]) == [["text"]]
    condition = apply("not", apply("not", Col("flag", BOOL)))
    assert str(backend.lower(condition).expr) == str(backend.column(Col("flag", BOOL)))


@pytest.mark.parametrize("backend", backend_params())
def test_static_branch_does_not_lower_dead_value(backend: Backend[Any]) -> None:
    dead = apply("__truediv__", lit(1), Col("value", INT))
    program = Program(lambda value: 7.0, {"value": INT}, If(TRUE, lit(7.0), dead, FLOAT))
    with patch.object(backend, "fallback", wraps=backend.fallback) as fallback:
        expr = backend.build(program)
    fallback.assert_not_called()
    assert backend.evaluate([expr], program.schema, [(0,)]) == [[7.0]]


@pytest.mark.parametrize("backend", backend_params())
def test_large_mixed_literal_keeps_guard(backend: Backend[Any]) -> None:
    program = trace(lambda value: value == 2**53 + 1, {"value": FLOAT})
    with patch.object(backend, "fallback", wraps=backend.fallback) as fallback:
        expr = backend.build(program)
    assert fallback.called
    rows = [(float(2**53),), (float(2**53 + 2),)]
    assert backend.evaluate([expr], program.schema, rows) == [program.run(rows)]
