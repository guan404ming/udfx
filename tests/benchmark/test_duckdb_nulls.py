from collections.abc import Callable
from pathlib import Path

import pyarrow
import pytest
from pyarrow import parquet

from bench.callables import null_boundary
from bench.runner import run
from bench.timing import Settings
from udfx.types import STR, Type
from workloads.models import Workload


@pytest.mark.parametrize(
    "function,input_type,result_type,values",
    [
        (lambda value: value.lower(), STR, STR, ["One", "Two"]),
        (
            lambda value: "missing" if value is None else value.lower(),
            STR.or_none(),
            STR,
            [None, "One"],
        ),
        (
            lambda value: None if value == "One" else value.lower(),
            STR,
            STR.or_none(),
            ["One", "Two"],
        ),
    ],
)
def test_duckdb_baselines_follow_declared_nullability(
    tmp_path: Path,
    function: Callable[..., object],
    input_type: Type,
    result_type: Type,
    values: list[str | None],
) -> None:
    path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.table({"value": values, "group": ["one"] * len(values)}), path)
    workload = Workload(
        "test",
        "nullability",
        function,
        {"value": input_type},
        result_type,
        path,
        {"value": "value"},
        category="group",
    )
    results = run(workload, ["duckdb"], Settings(warmups=0, rounds=1), tmp_path / "inputs")
    assert all(result["status"] in {"native", "guarded"} for result in results), results
    assert all(result["validation"] == "exact" for result in results)
    assert all(result["total_rows"] == len(values) for result in results)


def test_duckdb_mixed_input_nullability(tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    parquet.write_table(
        pyarrow.table(
            {"required": ["One", "Two"], "optional": [None, "!"], "group": ["one", "one"]}
        ),
        path,
    )
    workload = Workload(
        "test",
        "mixed_nullability",
        lambda required, optional: required.lower() + ("missing" if optional is None else optional),
        {"required": STR, "optional": STR.or_none()},
        STR,
        path,
        {"required": "required", "optional": "optional"},
        category="group",
    )
    results = run(workload, ["duckdb"], Settings(warmups=0, rounds=1), tmp_path / "inputs")
    assert all(result["status"] in {"native", "guarded"} for result in results), results
    assert all(result["validation"] == "exact" for result in results)


def test_null_boundary_preserves_nullable_signature() -> None:
    def function(value: str | None) -> str:
        return "missing" if value is None else value.lower()

    assert null_boundary(function, (STR.or_none(),)) is function


def test_null_boundary_preserves_valid_input_exceptions() -> None:
    def function(value: str) -> str:
        raise ValueError(value)

    wrapped = null_boundary(function, (STR,))
    assert wrapped(None) is None
    with pytest.raises(ValueError, match="failure"):
        wrapped("failure")
