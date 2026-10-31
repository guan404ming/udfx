from typing import Any
from unittest.mock import Mock

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.datafusion import DataFusionBackend
from udfx.backends.results import FallbackLowering
from udfx.backends.rules import Lowered
from udfx.frontend import trace
from udfx.ir import Col, lit
from udfx.program import Program
from udfx.testing import check_function
from udfx.types import BOOL, INT, INT64_MAX, STR


def parse(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def test_integer_recovery_keeps_one_unicode_conversion() -> None:
    backend = DataFusionBackend()
    program = trace(parse, {"text": STR}, result=INT.or_none())
    expr = backend.build(program)
    assert str(expr).count("translate(") == 1
    rows = [(value,) for value in ("12", "१२३", "1_000", "bad", str(INT64_MAX))]
    assert backend.evaluate([expr], program.schema, rows) == [program.run(rows)]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("intermediate", [False, True])
def test_integer_recovery_preserves_overflow(backend: Backend[Any], intermediate: bool) -> None:
    def selected(text: str) -> int | None:
        value = parse(text)
        return value + 1 if intermediate and value is not None else value

    rows = [(value,) for value in ("42", "१२३", "bad", str(INT64_MAX), str(INT64_MAX + 1))]
    assert check_function(selected, {"text": STR}, backend, rows) == []


@pytest.mark.parametrize(
    "backend", [value for value in backend_params() if value.id in {"duckdb", "datafusion"}]
)
def test_result_recovery_requires_exact_operands(backend: Backend[Any]) -> None:
    schema = {"value": INT, "exact": BOOL}
    program = Program(lambda value, exact: value if exact else value + 1, schema, Col("value", INT))
    recovery = Mock(side_effect=AssertionError("unguarded recovery"))
    lowered = Lowered(
        backend.column(Col("value", INT)),
        backend.column(Col("exact", BOOL)),
        backend.literal(lit(False)),
        recovery,
    )
    result = FallbackLowering(backend, program).finish(lowered)
    rows = [(10, True), (20, False)]
    assert backend.evaluate([result.expr], schema, rows) == [[10, 21]]
    recovery.assert_not_called()
