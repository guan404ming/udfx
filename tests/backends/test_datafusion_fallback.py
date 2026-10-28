from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pytest
from datafusion import udf

from udfx.backends.datafusion import DataFusionBackend
from udfx.ir import FALSE, Col, Lit
from udfx.program import Program
from udfx.types import BOOL, INT


@pytest.mark.parametrize("flags", [[True, True], [False, None], [True, False, None]])
def test_native_rows_skip_python_batches(
    monkeypatch: pytest.MonkeyPatch, flags: list[bool | None]
) -> None:
    received = []

    def register(function: Callable[..., Any], *arguments: Any) -> Any:
        def run(batch: pa.StructArray) -> pa.Array:
            received.extend(batch.field("value").to_pylist())
            return function(batch)

        return udf(run, *arguments)

    monkeypatch.setattr("udfx.backends.datafusion.backend.udf", register)
    backend = DataFusionBackend()
    schema = {"value": INT, "exact": BOOL.or_none()}
    program = Program(lambda value, exact: value * 2, schema, Col("value", INT))
    expr = backend.fallback(
        backend.literal(Lit(-1, INT)), backend.column(Col("exact", BOOL.or_none())), program
    )
    rows = list(enumerate(flags))
    assert backend.evaluate([expr], schema, rows) == [
        [-1 if exact else value * 2 for value, exact in rows]
    ]
    assert received == [value for value, exact in rows if not exact]


def test_fallback_without_arguments_preserves_row_count() -> None:
    backend = DataFusionBackend()
    program = Program(lambda: 9, {}, Lit(9, INT), deterministic=False)
    expr = backend.fallback(backend.literal(Lit(0, INT)), backend.literal(FALSE), program)
    assert backend.evaluate([expr], {}, [(), (), ()]) == [[9, 9, 9]]
