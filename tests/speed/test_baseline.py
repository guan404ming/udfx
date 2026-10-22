import importlib
from collections.abc import Callable
from typing import Any

import pyarrow
import pytest

from bench.speed.data import AttributeRow, caller, row_function
from udfx.backends import installed
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


def nested_attribute_row(row: Any) -> object:
    def value() -> object:
        def read() -> object:
            return row.value

        return read()

    assert type(row) is AttributeRow
    return value()


@pytest.mark.parametrize(
    "function", [dictionary_row, method_row, attribute_row, nested_attribute_row]
)
def test_row_callers_avoid_program_arguments(
    function: Callable[..., object], monkeypatch: pytest.MonkeyPatch
) -> None:
    program = Program(function, {"value": STR}, Col("value", STR), row_mode=True)
    monkeypatch.setattr(Program, "arguments", lambda *values: pytest.fail("udfx row adapter"))
    assert caller(program)("text") == "text"
    assert row_function(program)({"value": "text"}) == "text"


def test_dictionary_row_function_is_unchanged() -> None:
    program = Program(dictionary_row, {"value": STR}, Col("value", STR), row_mode=True)
    assert row_function(program) is dictionary_row


def test_row_caller_uses_schema_order() -> None:
    program = Program(
        lambda row: (row["second"], row["first"]),
        {"second": STR, "first": STR},
        Col("second", STR),
        row_mode=True,
    )
    assert caller(program)("left", "right") == ("left", "right")


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(backend.name, id=backend.name, marks=pytest.mark.xdist_group(backend.name))
        for backend in installed()
    ],
)
@pytest.mark.parametrize("function", [dictionary_row, nested_attribute_row])
def test_engine_row_baseline(engine: str, function: Callable[..., object]) -> None:
    program = Program(function, {"value": STR}, Col("value", STR), row_mode=True)
    values = ["text", "other"]
    table = pyarrow.table({"value": values})
    measurements = importlib.import_module(f"bench.speed.{engine}").measure(
        "row", program, table, values, 1
    )
    assert measurements
