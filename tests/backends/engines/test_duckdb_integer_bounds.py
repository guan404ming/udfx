import pytest

from udfx.backends.engines.duckdb import DuckDBBackend
from udfx.frontend import trace
from udfx.ir import Element, lit
from udfx.ops import apply
from udfx.semantics.bindings import bound_sources
from udfx.semantics.facts import integer_fits, integer_operation_fits
from udfx.testing import check_function
from udfx.testing.strategies import INT_TRAPS
from udfx.types import INT, INT64_MAX, INT64_MIN


def test_bounded_arithmetic_uses_native_int64() -> None:
    def bounded(value: int) -> int:
        return (value % 100 + 2) * 3 - 1

    backend = DuckDBBackend()
    schema = {"value": INT.or_none()}
    program = trace(bounded, schema)
    lowered = backend.lower_program(program)
    assert "HUGEINT" not in str(lowered.expr)
    rows = [(value,) for value in (*INT_TRAPS, None)]
    assert check_function(bounded, schema, backend, rows) == []


@pytest.mark.parametrize("nested", [False, True])
def test_wide_operand_placeholder_keeps_native_arithmetic_safe(nested: bool) -> None:
    value = apply("__sub__", lit(INT64_MIN - 1, INT), lit(INT64_MIN, INT))
    if nested:
        value = apply("__sub__", apply("max", value, lit(-1)), lit(INT64_MIN, INT))
    assert integer_fits(value, size_limit=INT64_MAX)
    assert not integer_operation_fits(value, size_limit=INT64_MAX)
    backend = DuckDBBackend()
    lowered = backend.lower(value)
    assert "HUGEINT" in str(lowered.expr)
    _, exact = backend.evaluate([lowered.expr, backend.exact_expression(lowered)], {}, [()])
    assert exact == [False]


def test_integer_operation_proof_resolves_bound_dependencies() -> None:
    operand = Element("operand", INT)
    source = apply("__sub__", lit(INT64_MIN - 1, INT), lit(INT64_MIN, INT))
    value = apply("__sub__", operand, lit(INT64_MIN, INT))
    with bound_sources({operand: source}):
        assert integer_fits(value, size_limit=INT64_MAX)
        assert not integer_operation_fits(value, size_limit=INT64_MAX)
