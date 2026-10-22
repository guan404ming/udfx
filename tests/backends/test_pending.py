from collections.abc import Sequence
from typing import Any, cast

import pyarrow as pa
import pytest

from udfx.backends import arrow
from udfx.backends.base import Backend
from udfx.backends.duckdb import DuckDBBackend
from udfx.ir import Col
from udfx.program import Program
from udfx.types import BOOL, INT


@pytest.mark.parametrize("flags", [[True, True, True], [True, False, None], [False, False, False]])
def test_duckdb_case_sends_only_pending_rows_to_arrow(
    flags: list[bool | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    batches: list[list[int]] = []
    original = arrow.fallback

    def record(
        backend: Backend[Any],
        program: Program,
        flags: pa.Array,
        native: pa.Array,
        columns: Sequence[pa.Array],
    ) -> pa.Array:
        batches.append(cast(list[int], columns[0].to_pylist()))
        return original(backend, program, flags, native, columns)

    monkeypatch.setattr(arrow, "fallback", record)
    backend = DuckDBBackend()
    schema = {"value": INT, "flag": BOOL.or_none()}
    program = Program(lambda value, flag: value * 10, schema, Col("value", INT))
    expr = backend.fallback(
        backend.column(Col("value", INT)), backend.column(Col("flag", BOOL.or_none())), program
    )
    rows = list(zip([1, 2, 3], flags, strict=True))
    assert backend.evaluate([expr], schema, rows) == [
        [value if flag else value * 10 for value, flag in rows]
    ]
    assert sorted(value for batch in batches for value in batch) == [
        value for value, flag in rows if not flag
    ]
    if all(flags):
        assert batches == []
