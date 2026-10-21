from unittest.mock import patch

import pyarrow as pa
import pytest

from udfx.backends import arrow
from udfx.backends.polars import PolarsBackend
from udfx.ir import Col
from udfx.program import Program
from udfx.types import INT, STR, Type, list_of


@pytest.mark.parametrize("values", [[], [1, None, 3]])
def test_exact_batch_keeps_native_array(values: list[int | None]) -> None:
    backend = PolarsBackend()
    program = Program(lambda value: value, {"value": INT.or_none()}, Col("value", INT.or_none()))
    native = pa.array(values, pa.int64())
    flags = pa.array([True] * len(values), pa.bool_())
    with patch.object(backend, "run_fallback_rows") as run:
        assert arrow.fallback(backend, program, flags, native, [native]) is native
    run.assert_not_called()


@pytest.mark.parametrize(
    ("type", "values"),
    [(INT.or_none(), [1, None, 3, 4]), (list_of(STR.or_none()), [["a"], [], [None], ["d"]])],
)
def test_fallback_reads_only_pending_rows(type: Type, values: list[object]) -> None:
    backend = PolarsBackend()
    observed: list[object] = []

    def identify(value: object) -> object:
        observed.append(value)
        return value

    program = Program(identify, {"value": type}, Col("value", type))
    columns = pa.array(values, arrow.dtype(type))
    native = pa.array([values[3], None, None, values[0]], arrow.dtype(type))
    flags = pa.array([True, False, None, True])
    result = arrow.fallback(backend, program, flags, native, [columns])
    assert observed == [values[1], values[2]]
    assert result.to_pylist() == [values[3], values[1], values[2], values[0]]
