import json
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import lit
from udfx.testing import check_function
from udfx.types import STR, Type, list_of


@pytest.mark.parametrize(
    "backend",
    [
        value
        for value in backend_params()
        if value.id in {"duckdb", "datafusion", "spark", "spark-ansi-off"}
    ],
)
@pytest.mark.parametrize(
    "function",
    [
        lambda left, right: len(set(left) & set(right)) / len(set(left) | set(right)),
        lambda left, right: float(len(set(left) & set(right))) / len(set(left) | set(right)),
    ],
)
def test_disjoint_ratio_uses_empty_intersection(
    backend: Backend[Any], function: Callable[..., float], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = backend.select
    observed = []
    zero = str(backend.literal(lit(0.0)))

    def select(condition: Any, then: Any, orelse: Any) -> Any:
        if str(then) == zero:
            observed.append(condition)
        return original(condition, then, orelse)

    monkeypatch.setattr(backend, "select", select)
    schema: dict[str, Type] = dict.fromkeys(("left", "right"), list_of(STR).or_none())
    rows = [
        (left, right)
        for left in (None, [], ["a", "a", "b"], ["é", "中", ""])
        for right in (None, [], ["c"], ["a", "é", "中"])
    ]
    assert check_function(function, schema, backend, rows) == []
    assert observed


@pytest.mark.parametrize(
    "backend",
    [
        value
        for value in backend_params()
        if value.id in {"duckdb", "datafusion", "spark", "spark-ansi-off"}
    ],
)
def test_disjoint_ratio_preserves_json_guards(backend: Backend[Any]) -> None:
    def ratio(left: str, right: str) -> float | None:
        try:
            first = set(json.loads(left))
            second = set(json.loads(right))
            return float(len(first & second)) / len(first | second)
        except (ValueError, TypeError, ZeroDivisionError):
            return None

    schema: dict[str, Type] = dict.fromkeys(("left", "right"), STR.or_none())
    rows = [
        (None, "[]"),
        ("[", "[]"),
        ("[]", "[]"),
        ('["a"]', '["b"]'),
        ('["a", "b"]', '["a", "c"]'),
        ('["a", "a"]', '["a"]'),
        ('["é", "中"]', '["中"]'),
        ('["a"]', "[1]"),
        ("[1, 2]", "[2, 3]"),
        ("[true, 1]", "[1]"),
        ("[null]", "[null]"),
        ("{}", "[]"),
        ('{"a": 1}', '["a"]'),
        ('"ab"', '["b"]'),
        ("[[]]", "[]"),
        ("null", "[]"),
    ]
    assert check_function(ratio, schema, backend, rows) == []
    program = trace(ratio, schema)
    lowered = backend.lower_program(program)
    assert backend.evaluate([backend.exact_expression(lowered)], schema, rows[2:7]) == [[True] * 5]
