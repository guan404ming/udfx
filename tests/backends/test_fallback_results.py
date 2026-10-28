from itertools import product
from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.diagnostics import capture_lowering
from udfx.backends.results import FallbackLowering
from udfx.ir import Col, Guarded, If, lit
from udfx.ops import apply
from udfx.program import Program
from udfx.types import BOOL, INT, INT64_MAX, Type


@pytest.mark.parametrize(
    "backend", [value for value in backend_params() if value.id in {"duckdb", "datafusion"}]
)
@pytest.mark.parametrize("nested_then", [False, True])
def test_guarded_cases_share_one_pending_fallback(backend: Backend[Any], nested_then: bool) -> None:
    schema: dict[str, Type] = dict.fromkeys(("first", "valid", "second", "exact"), BOOL.or_none())
    first, valid, second, exact = (Col(name, type) for name, type in schema.items())
    inner = If(Guarded(second, exact), lit(10), lit(20), INT)
    value = If(
        Guarded(first, valid),
        inner if nested_then else lit(30),
        lit(30) if nested_then else inner,
        INT,
    )

    def reference(first: bool, valid: bool, second: bool, exact: bool) -> int:
        if bool(first) != nested_then:
            return 30 if valid else 31
        return (10 if exact else 11) if second else 20

    program = Program(reference, schema, value)
    with (
        patch.object(backend, "fallback", wraps=backend.fallback) as fallback,
        patch.object(backend, "select_many", wraps=backend.select_many) as select,
        capture_lowering() as details,
    ):
        expr = backend.build(program)
    fallback.assert_called_once()
    select.assert_called_once()
    rows = list(product((True, False, None), repeat=4))
    assert backend.evaluate([expr], schema, rows) == [program.run(rows)]
    assert backend.evaluate([fallback.call_args.args[1]], schema, rows) == [[False] * len(rows)]
    assert details[-1].exact is not None
    native = backend.evaluate([details[-1].exact], schema, rows)[0]
    assert [flag is True for flag in native] == [
        valid is True and (exact is True if bool(first) == nested_then else True)
        for first, valid, _second, exact in rows
    ]


@pytest.mark.parametrize("backend", [value for value in backend_params() if value.id == "duckdb"])
def test_guarded_condition_keeps_native_overflow(backend: Backend[Any]) -> None:
    schema = {"active": BOOL, "valid": BOOL, "value": INT}
    condition = Guarded(Col("active", BOOL), Col("valid", BOOL))
    value = If(condition, apply("__add__", Col("value", INT), lit(1)), lit(0), INT)
    program = Program(lambda active, valid, value: value + 1 if active else 0, schema, value)
    expr = backend.build(program)
    with patch("udfx.backends.arrow.fallback", side_effect=AssertionError("unexpected fallback")):
        assert backend.evaluate([expr], schema, [(False, True, INT64_MAX)]) == [[0]]
        with pytest.raises(OverflowError, match="does not fit int"):
            backend.evaluate([expr], schema, [(True, True, INT64_MAX)])


@pytest.mark.parametrize(
    "backend", [value for value in backend_params() if value.id in {"duckdb", "datafusion"}]
)
def test_guarded_cases_keep_bound_branches(backend: Backend[Any]) -> None:
    schema = {"active": BOOL, "valid": BOOL}
    active = Col("active", BOOL)
    inner = If(active, lit(10), lit(20), INT)
    value = If(Guarded(active, Col("valid", BOOL)), inner, lit(30), INT)
    program = Program(lambda active, valid: 7 if active else 30, schema, value)
    results = FallbackLowering(backend, program)
    lowered = results.lower(value, {inner: backend.literal(lit(7))})
    assert backend.evaluate([lowered.expr], schema, [(True, True), (False, True)]) == [[7, 30]]
