import math
from collections.abc import Callable
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.ir import TRUE, Lit
from udfx.program import Program
from udfx.types import BOOL, DATETIME, FLOAT, INT, INT64_MAX, INT64_MIN, STR, Type, list_of


def integer(value: Any) -> int:
    return int(value)


def length(value: Any) -> int:
    return len(value)


def total(value: Any) -> int:
    return sum(value)


FAULTS = [
    (lambda value: 1 / value, INT, 0),
    (integer, STR, "bad\x00\n'\\😀udfx fault "),
    (integer, FLOAT, float("nan")),
    (integer, FLOAT, float("inf")),
    (integer, FLOAT, float(INT64_MAX)),
    (integer, FLOAT, -1e100),
    (integer, STR, str(INT64_MAX + 1)),
    (length, STR.or_none(), None),
    (lambda value: value.lower(), STR.or_none(), None),
    (lambda value: value[1], STR, ""),
    (lambda value: {"known": 1}[value], STR, "'\n😀"),
    (lambda value: value + 1, INT, INT64_MAX),
    (lambda value: value - 1, INT, INT64_MIN),
    (lambda value: value * 2, INT, INT64_MAX),
    (lambda value: value // -1, INT, INT64_MIN),
    (total, list_of(INT), [INT64_MAX, 1]),
]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(("function", "type", "value"), FAULTS)
def test_fault_preserves_exception_message(
    backend: Backend[Any], function: Callable[..., object], type: Type, value: object
) -> None:
    schema = {"value": type}
    translation = compile(function, schema, backend)
    assert translation.program is not None, translation.unsupported
    with pytest.raises(Exception) as expected:
        translation.program.run([(value,)])
    with pytest.raises(expected.type) as actual:
        backend.evaluate([translation.expr], schema, [(value,)])
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("backend", backend_params())
def test_fault_payload_restores_all_input_types(backend: Backend[Any]) -> None:
    schema = {
        "none": STR.or_none(),
        "nan": FLOAT,
        "negative_zero": FLOAT,
        "minimum": INT,
        "maximum": INT,
        "text": STR,
        "flag": BOOL,
        "date": DATETIME,
        "dates": list_of(DATETIME.or_none()),
        "numbers": list_of(FLOAT.or_none()),
        "strings": list_of(STR.or_none()),
    }
    row = (
        None,
        float("nan"),
        -0.0,
        INT64_MIN,
        INT64_MAX,
        "\x00\n\r\"'\\😀udfx fault ",
        True,
        datetime.min,
        [None, datetime.max],
        [None, float("nan"), -0.0, float("inf"), 5e-324],
        [None, "", "\x00\n😀"],
    )

    def fail(*values: Any) -> int:
        restored = list(values)
        restored[1] = math.isnan(restored[1])
        restored[2] = math.copysign(1.0, restored[2])
        restored[9] = [
            None if item is None else (math.isnan(item), math.copysign(1.0, item), repr(item))
            for item in restored[9]
        ]
        raise ValueError(repr(restored))

    program = Program(fail, schema, Lit(0, INT), TRUE)
    with pytest.raises(ValueError) as expected:
        program.run([row])
    with pytest.raises(ValueError) as actual:
        backend.evaluate([backend.build(program)], schema, [row])
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("function", "type"),
    [
        (lambda value: value + 1, INT),
        (lambda value: 1.0 / value, FLOAT),
        (integer, STR),
        (total, list_of(INT)),
    ],
)
def test_native_fault_plan_has_no_fallback(
    backend: Backend[Any], function: Callable[..., object], type: Type
) -> None:
    if backend.name not in {"spark", "duckdb"}:
        pytest.skip("engine has no native raise")
    with patch.object(backend, "fallback", wraps=backend.fallback) as fallback:
        translation = compile(function, {"value": type}, backend)
    assert translation.ok, translation.unsupported
    assert not fallback.called
    assert translation.expr is not None
    if isinstance(backend, SparkBackend):
        session = backend.session()
        value = "array(id)" if type.item is not None else "id"
        plan = session.range(1).selectExpr(f"{value} AS value").select(translation.expr)
        explanation = StringIO()
        with redirect_stdout(explanation):
            plan.explain()
        assert "Python" not in explanation.getvalue()


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: (value + 1) - 1,
        lambda value: (value * 2) // 2,
        lambda value: int(str(value) + "0") // 10,
        lambda value: value // (INT64_MAX + 1) + value,
    ],
)
def test_intermediate_overflow_keeps_fallback(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    translation = compile(function, {"value": INT}, backend)
    assert translation.ok, translation.unsupported
    assert backend.evaluate([translation.expr], {"value": INT}, [(INT64_MAX,)]) == [[INT64_MAX]]


@pytest.mark.parametrize("backend", backend_params())
def test_native_fault_without_columns(backend: Backend[Any]) -> None:
    program = Program(lambda: 1 // 0, {}, Lit(0, INT), TRUE)
    expr = backend.build(program)
    assert backend.evaluate([expr], {}, []) == [[]]
    with pytest.raises(ZeroDivisionError, match="integer division or modulo by zero"):
        backend.evaluate([expr], {}, [()])


@pytest.mark.parametrize("backend", backend_params())
def test_conditional_overflow_only_raises_in_the_active_branch(backend: Backend[Any]) -> None:
    schema = {"value": INT, "active": BOOL}
    translation = compile(lambda value, active: value + 1 if active else value, schema, backend)
    assert backend.evaluate([translation.expr], schema, [(INT64_MAX, False), (0, True)]) == [
        [INT64_MAX, 1]
    ]
    with pytest.raises(OverflowError, match="does not fit int"):
        backend.evaluate([translation.expr], schema, [(INT64_MAX, True)])
