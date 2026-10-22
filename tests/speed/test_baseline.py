import importlib
from collections.abc import Callable
from typing import Any

import pyarrow
import pytest
from conftest import engine_names

from bench.speed.data import AttributeRow, MixedRow, caller, row_function, row_type
from udfx.ir import Col
from udfx.program import Program
from udfx.types import STR


def dictionary_row(row: dict[str, object]) -> object:
    assert type(row) is dict
    return row["value"]


def method_row(row: dict[str, str]) -> str:
    assert type(row) is dict
    return row["value"].strip()


def attribute_row(row: Any) -> object:
    assert type(row) is AttributeRow
    return row.value


def positional_row(row: tuple[object, ...]) -> object:
    assert type(row) is tuple
    return row[0]


def negative_positional_row(row: tuple[object, ...]) -> object:
    assert type(row) is tuple
    return row[-1]


def nested_positional_row(row: tuple[object, ...]) -> object:
    def value() -> object:
        return row[-1]

    assert type(row) is tuple
    return value()


def name_attribute_row(row: Any) -> object:
    assert type(row) is AttributeRow
    assert row["value"] == row.value
    return row.value


def positional_name_row(row: Any) -> object:
    assert type(row) is MixedRow
    assert row[0] == row["value"]
    return row[-1]


def positional_attribute_row(row: Any) -> object:
    assert type(row) is MixedRow
    assert row[0] == row.value
    return row[-1]


def mixed_row(row: Any) -> object:
    assert type(row) is MixedRow
    assert row[0] == row["value"] == row.value
    return row[-1]


def nested_attribute_row(row: Any) -> object:
    def value() -> object:
        def read() -> object:
            return row.value

        return read()

    assert type(row) is AttributeRow
    return value()


@pytest.mark.parametrize(
    "function",
    [
        dictionary_row,
        method_row,
        attribute_row,
        nested_attribute_row,
        positional_row,
        negative_positional_row,
        nested_positional_row,
        name_attribute_row,
        positional_name_row,
        positional_attribute_row,
        mixed_row,
    ],
)
def test_row_callers_avoid_program_arguments(
    function: Callable[..., object], monkeypatch: pytest.MonkeyPatch
) -> None:
    program = Program(function, {"value": STR}, Col("value", STR), row_mode=True)
    monkeypatch.setattr(Program, "arguments", lambda *values: pytest.fail("udfx row adapter"))
    assert caller(program)("text") == "text"
    row = ("text",) if row_type(program) is tuple else {"value": "text"}
    assert row_function(program)(row) == "text"


@pytest.mark.parametrize("function", [dictionary_row, positional_row])
def test_builtin_row_function_is_unchanged(function: Callable[..., object]) -> None:
    program = Program(function, {"value": STR}, Col("value", STR), row_mode=True)
    assert row_function(program) is function


@pytest.mark.parametrize(
    "function",
    [
        lambda row: (row["second"], row["first"]),
        lambda row: (row[0], row[-1]),
        lambda row: (row[0], row["first"]),
        lambda row: (row.second, row[-1]),
    ],
)
def test_row_caller_uses_schema_order(function: Callable[..., object]) -> None:
    program = Program(
        function,
        {"second": STR, "first": STR},
        Col("second", STR),
        row_mode=True,
    )
    assert caller(program)("left", "right") == ("left", "right")


@pytest.mark.parametrize(
    "function", [lambda row: row["value"][0], lambda row: row["value"].strip()]
)
def test_column_accesses_keep_dictionary_rows(function: Callable[..., object]) -> None:
    program = Program(function, {"value": STR, "strip": STR}, Col("value", STR), row_mode=True)
    assert row_type(program) is dict


@pytest.mark.parametrize(
    "engine",
    [pytest.param(name, id=name, marks=pytest.mark.xdist_group(name)) for name in engine_names()],
)
@pytest.mark.parametrize(
    "function",
    [
        dictionary_row,
        nested_attribute_row,
        negative_positional_row,
        positional_name_row,
        positional_attribute_row,
        mixed_row,
    ],
)
def test_engine_row_baseline(engine: str, function: Callable[..., object]) -> None:
    program = Program(function, {"value": STR}, Col("value", STR), row_mode=True)
    values = ["text", "other"]
    table = pyarrow.table({"value": values})
    measurements = importlib.import_module(f"bench.speed.{engine}").measure(
        "row", program, table, values, 1
    )
    assert measurements
    if engine == "polars":
        form = "map_rows" if function is negative_positional_row else "map_elements"
        assert all(measurement.form == form for measurement in measurements)
