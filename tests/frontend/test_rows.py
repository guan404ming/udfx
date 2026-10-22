# ruff: noqa: PLW0108
from collections.abc import Callable
from dataclasses import replace
from types import FunctionType
from typing import Any
from unittest.mock import Mock

import pytest

from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import FALSE, Col, lit
from udfx.ops import apply
from udfx.types import FLOAT, INT, STR, Type


def total(row: Any) -> float:
    return row["price"] * row.quantity


def nested(row: Any) -> str:
    def clean(value: str) -> str:
        def lower(text: str) -> str:
            return text.lower()

        return lower(value.strip())

    return clean(row["image"])


def pass_row(row: Any) -> str:
    def read(value: Any) -> str:
        return value["image"]

    return read(row)


def pass_nested_row(row: Any) -> str:
    suffix = "!"

    def read(record: Any) -> str:
        def clean(value: Any) -> str:
            return value.image.strip().lower() + suffix

        return clean(value=record)

    return read(row)


def iterate(row: Any) -> str:
    for value in row:
        return value
    return ""


def length(row: Any) -> int:
    return len(row)


def text(row: Any) -> str:
    return str(row)


def assign_key(row: Any) -> str:
    row["image"] = "changed"
    return row["image"]


def assign_attribute(row: Any) -> str:
    row.image = "changed"
    return row.image


def assign_alias(row: Any) -> str:
    other = row
    return other.image


def test_row_reads_become_columns() -> None:
    schema = {"quantity": INT, "price": FLOAT}
    program = trace(total, schema)
    assert program.row_mode
    assert program.schema == schema
    assert program.value == apply("__mul__", Col("price", FLOAT), Col("quantity", INT))
    assert program.fault == FALSE
    assert program.run([(3, 2.5), (4, 1.5)]) == [7.5, 6.0]


@pytest.mark.parametrize(
    ("function", "schema", "values", "name"),
    [
        (lambda row: row["image"], {"image": STR}, ("photo",), "image"),
        (lambda row: row[0], {"image": STR, "count": INT}, ("photo", 2), "image"),
        (lambda row: row[1], {"count": INT, "image": STR}, (2, "photo"), "image"),
        (lambda row: row[-1], {"count": INT, "image": STR}, (2, "photo"), "image"),
        (lambda row: row[-2], {"image": STR, "count": INT}, ("photo", 2), "image"),
        (lambda row: row.image, {"image": STR}, ("photo",), "image"),
        (lambda row: row["row"], {"row": STR, "unused": INT}, ("photo", 1), "row"),
        (lambda row: row["first name"], {"first name": STR}, ("photo",), "first name"),
        (lambda row: row._values, {"_values": STR}, ("photo",), "_values"),
        (lambda row: row.__getitem__, {"__getitem__": STR}, ("photo",), "__getitem__"),
        (lambda row: row.__class__, {"__class__": STR}, ("photo",), "__class__"),
        (lambda row: row.get, {"get": STR}, ("photo",), "get"),
    ],
)
def test_row_access(
    function: Callable[..., object], schema: dict[str, Type], values: tuple[object, ...], name: str
) -> None:
    program = trace(function, schema)
    assert program.value == Col(name, STR)
    assert program.run([values]) == ["photo"]


def test_row_column_passes_through_nested_helpers() -> None:
    program = trace(nested, {"image": STR})
    assert program.value == apply(".strip", apply(".lower", Col("image", STR)))
    assert program.run([(" PHOTO ",)]) == ["photo"]


@pytest.mark.parametrize("function", [pass_row, lambda row: pass_row(row)])
def test_row_passes_to_helpers(function: Callable[..., object]) -> None:
    program = trace(function, {"image": STR})
    assert program.value == Col("image", STR)
    assert program.run([("photo",)]) == ["photo"]


def test_row_passes_through_nested_closures() -> None:
    program = trace(pass_nested_row, {"image": STR})
    assert program.value == apply(
        "__add__", apply(".strip", apply(".lower", Col("image", STR))), lit("!")
    )
    assert program.run([(" PHOTO ",)]) == ["photo!"]


@pytest.mark.parametrize("key", ["image", 0, -1])
@pytest.mark.parametrize("helper", [False, True])
def test_local_constant_row_key(key: str | int, helper: bool) -> None:
    def read(row: Any) -> str:
        feature = key
        return row[feature]

    program = trace((lambda row: read(row)) if helper else read, {"image": STR})
    assert program.value == Col("image", STR)
    assert program.run([("photo",)]) == ["photo"]


def test_fallback_helper_receives_original_row() -> None:
    program = trace(lambda row: pass_row(row), {"image": STR})
    assert isinstance(program.function, FunctionType)
    helper = Mock(wraps=program.function.__globals__["pass_row"])
    program.function.__globals__["pass_row"] = helper
    function = Mock(wraps=program.function)
    program = replace(program, function=function)
    assert program.run([("photo",)]) == ["photo"]
    assert helper.call_args.args[0] is function.call_args.args[0]


def test_matching_parameter_stays_scalar() -> None:
    program = trace(lambda image: image.lower(), {"image": STR})
    assert not program.row_mode
    assert program.run([("PHOTO",)]) == ["photo"]


def test_row_with_no_columns() -> None:
    program = trace(lambda row: 7, {})
    assert program.row_mode
    assert program.run([(), ()]) == [7, 7]


def test_multiple_parameters_still_require_matching_schema() -> None:
    with pytest.raises(Unsupported, match="does not match parameters"):
        trace(lambda first, second: first + second, {"image": STR})


@pytest.mark.parametrize("function", [lambda row: row["missing"], lambda row: row.missing])
def test_unknown_row_column_names_the_key(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported, match="unknown row column 'missing'"):
        trace(function, {"image": STR})


@pytest.mark.parametrize("index", [1, -2])
def test_row_position_out_of_range(index: int) -> None:
    def caught(row: Any) -> str:
        try:
            return row[index]
        except IndexError:
            return "missing"

    program = trace(caught, {"image": STR})
    assert program.value == lit("missing")
    assert program.fault == FALSE
    assert program.run([("photo",)]) == ["missing"]

    def uncaught(row: Any) -> str:
        return row[index]

    with pytest.raises(IndexError):
        uncaught(*program.arguments(("photo",)))

    with pytest.raises(Unsupported, match="function never returns"):
        trace(uncaught, {"image": STR})


@pytest.mark.parametrize(
    ("function", "reason"),
    [
        (length, "passing a row to a call"),
        (text, "passing a row to a call"),
        (lambda row: pow(base=row, exp=2), "passing a row to a call"),
        (lambda row: row(), "cannot call a row"),
        (iterate, "loops and comprehensions"),
        (lambda row: row == "image", "expected a value, got a row"),
        (lambda row: row < "image", "expected a value, got a row"),
        (lambda row: row is None, "None test of a row"),
        (lambda row: "image" in row, "expected a value, got a row"),
        (lambda row: row + "image", "expected a value, got a row"),
        (lambda row: not row, "truth of a row"),
        (lambda row: row[row.image], "constant string or integer key"),
        (lambda row: row[:1], "expected a value, got a row"),
        (lambda row: row.get("image"), "unknown row column 'get'"),
        (assign_key, "STORE_SUBSCR"),
        (assign_attribute, "STORE_ATTR"),
        (assign_alias, "assignment of a row"),
        (lambda row: row, "expected a value, got a row"),
        (lambda row: (row,), "tuple returns are unsupported"),
    ],
)
@pytest.mark.parametrize("helper", [False, True])
def test_other_row_uses_are_unsupported(
    function: Callable[..., object], reason: str, helper: bool
) -> None:
    with pytest.raises(Unsupported, match=reason):
        trace((lambda row: function(row)) if helper else function, {"image": STR})
