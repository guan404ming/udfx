from dataclasses import replace
from pathlib import Path
from typing import cast

import pyarrow
import pytest
from pyarrow import parquet

from bench.data import expected_query, prepare
from bench.validation import validate
from udfx.types import FLOAT, INT, STR
from workloads.flights import load
from workloads.models import Join, Workload
from workloads.storage import FIXTURES


def test_query_oracle_includes_join_filter_and_counts(tmp_path: Path) -> None:
    workload = load(FIXTURES)[0]
    data = prepare(workload, tmp_path, None)
    expected = expected_query(data)
    assert sum(cast(int, row[2]) for row in expected) == 32
    assert len(expected) < 32
    assert all(isinstance(row[0], str) for row in expected)
    validate(
        pyarrow.table(
            dict(zip(["category", "value", "count"], zip(*expected, strict=True), strict=True))
        ),
        data,
        "query",
        "oracle",
    )


def test_source_rows_are_not_repeated_or_filtered(tmp_path: Path) -> None:
    workload = load(FIXTURES)[1]
    data = prepare(workload, tmp_path, 100000)
    assert data.table.num_rows == 32
    assert data.metadata["rows"] == 32
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


def test_missing_function_and_empty_data_fail(tmp_path: Path) -> None:
    workload = load(FIXTURES)[0]
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


def test_lookup_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "lookup.parquet"
    parquet.write_table(pyarrow.table({"airport": ["ATL", "ATL"], "city": ["A", "B"]}), path)
    workload = replace(load(FIXTURES)[0], join=Join(path, "origin", "airport", "city"))
    with pytest.raises(ValueError, match="unique"):
        prepare(workload, tmp_path / "inputs", None)
