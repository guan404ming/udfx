from datetime import date
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.ir import FALSE, Col, Lit
from udfx.program import Program
from udfx.testing.datetime_traps import DATE_TRAPS
from udfx.types import DATE


def python_date(value: date | None) -> date | None:
    if value is not None and type(value) is not date:
        raise TypeError(type(value).__name__)
    return value


@pytest.mark.parametrize("backend", backend_params())
def test_date_transport_and_fallback(backend: Backend[Any]) -> None:
    schema = {"value": DATE.or_none()}
    column = Col("value", DATE.or_none())
    program = Program(python_date, schema, column)
    native = backend.column(column)
    patched = backend.fallback(native, backend.literal(FALSE), program)
    values = [*DATE_TRAPS, None]
    outputs = backend.evaluate(
        [native, patched, backend.literal(Lit(date.min, DATE))],
        schema,
        [(value,) for value in values],
    )
    assert outputs == [values, values, [date.min] * len(values)]
    assert all(value is None or type(value) is date for column in outputs for value in column)
