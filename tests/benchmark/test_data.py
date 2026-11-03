from dataclasses import replace
from pathlib import Path
from typing import cast

import pyarrow
import pytest
from pyarrow import parquet

from bench.data import expected_query, prepare
from bench.validation import validate
from tests.benchmark.samples import ROWS, load
from udfx.types import FLOAT, INT, STR
from workloads.models import Workload


def test_query_oracle_filters_and_counts(tmp_path: Path, sample: Path) -> None:
    workload = load(sample)[0]
    data = prepare(workload, tmp_path, None)
    expected = expected_query(data)
    assert sum(cast(int, row[2]) for row in expected) == sum(
        value is not None for value in data.expected
    )
    assert len(expected) < ROWS
    assert all(isinstance(row[0], str) for row in expected)
    validate(
        pyarrow.table(
            dict(zip(["category", "value", "count"], zip(*expected, strict=True), strict=True))
        ),
        data,
        "query",
        "oracle",
    )


def test_source_rows_are_not_repeated_or_filtered(tmp_path: Path, sample: Path) -> None:
    workload = load(sample)[1]
    data = prepare(workload, tmp_path, 100000)
    assert data.table.num_rows == ROWS
    assert data.metadata["rows"] == ROWS
    assert data.metadata["row_selection"] == "source prefix"
    assert prepare(workload, tmp_path, 3).table.num_rows == 3


def test_oracle_exception_stops_instead_of_dropping_rows(tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.table({"value": [1, 0, 2]}), path)
    workload = Workload(
        "test", "divide", lambda value: 1 / value, {"value": INT}, FLOAT, path, {"value": "value"}
    )
    with pytest.raises(ValueError, match="ZeroDivisionError at input row 1"):
        prepare(workload, tmp_path / "inputs", None)


def test_missing_function_and_empty_data_fail(tmp_path: Path, sample: Path) -> None:
    workload = load(sample)[0]
    with pytest.raises(ValueError, match="Missing Python"):
        prepare(replace(workload, function=None), tmp_path, None)
    with pytest.raises(ValueError, match="no input rows"):
        prepare(workload, tmp_path, 0)


def test_constant_arguments_keep_schema_order(tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    parquet.write_table(pyarrow.table({"input": ["a", None]}), path)
    workload = Workload(
        "test",
        "constant",
        lambda first, second: second if first is None else first,
        {"first": STR.or_none(), "second": STR},
        STR,
        path,
        {"first": "input"},
        constants={"second": "fallback"},
    )
    data = prepare(workload, tmp_path / "inputs", None)
    assert data.expected == ["a", "fallback"]
    assert expected_query(data) == [[0, "a"], [1, "fallback"]]
    validate(
        pyarrow.table({"position": [1, 0], "value": ["fallback", "a"]}), data, "memory", "reordered"
    )
    with pytest.raises(AssertionError, match="rows !="):
        validate(pyarrow.table({"position": [0], "value": ["a"]}), data, "memory", "missing")
    with pytest.raises(AssertionError):
        validate(
            pyarrow.table({"position": [0, 1], "value": ["wrong", "a"]}), data, "memory", "wrong"
        )
