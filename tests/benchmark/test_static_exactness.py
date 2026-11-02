import math
from pathlib import Path
from typing import Any

import pyarrow
import pytest
from conftest import backend_params
from pyarrow import parquet

from bench.inspection import pending_share
from bench.runner import ENGINES, run
from bench.timing import Settings
from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Col
from udfx.types import FLOAT
from workloads.models import Workload


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("rows", [[], [(1.0,), (4.0,)]])
def test_pending_share_normalizes_static_false(
    backend: Backend[Any], rows: list[tuple[float]]
) -> None:
    program = trace(lambda value: math.log(value, 2.0), {"value": FLOAT})
    assert backend.lower_program(program).exact is False
    assert pending_share(
        backend,
        program,
        lambda expr: backend.evaluate(
            [backend.column(Col("value", FLOAT)), expr], program.schema, rows
        )[1],
    ) == (1.0 if rows else 0.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_benchmark_reports_static_false(engine: str, tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.table({"value": [1.0, 2.0, 4.0]}), source)
    workload = Workload(
        "test",
        "static_false",
        lambda value: math.log(value, 2.0),
        {"value": FLOAT},
        FLOAT,
        source,
        {"value": "value"},
    )
    results = run(workload, [engine], Settings(warmups=0, rounds=1), tmp_path / "inputs")
    assert len(results) == 2
    for result in results:
        assert result["status"] == "python", result.get("reason")
        assert result["native_rows"] == 0
        assert result["native_row_fraction"] == 0.0
        assert result["compiled_udfs"] == 0
        assert result["comparisons"]
