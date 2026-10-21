from collections.abc import Callable
from typing import Any

import pytest

from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import FALSE, Col
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
    assert program.value == apply(".lower", apply(".strip", Col("image", STR)))
    assert program.run([(" PHOTO ",)]) == ["photo"]


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


@pytest.mark.parametrize(
    ("function", "reason"),
    [
        (pass_row, "passing a row to a call"),
        (length, "passing a row to a call"),
        (text, "passing a row to a call"),
        (lambda row: row(), "cannot call a row"),
        (iterate, "loops and comprehensions"),
        (lambda row: row == "image", "expected a value, got a row"),
        (lambda row: row < "image", "expected a value, got a row"),
        (lambda row: row is None, "None test of a row"),
        (lambda row: "image" in row, "expected a value, got a row"),
        (lambda row: row + "image", "expected a value, got a row"),
        (lambda row: not row, "truth of a row"),
        (lambda row: row[row.image], "constant string key"),
        (lambda row: row[0], "constant string key"),
        (lambda row: row[:1], "expected a value, got a row"),
        (lambda row: row.get("image"), "unknown row column 'get'"),
        (assign_key, "STORE_SUBSCR"),
        (assign_attribute, "STORE_ATTR"),
        (assign_alias, "assignment of a row"),
        (lambda row: row, "expected a value, got a row"),
        (lambda row: (row,), "expected a value, got a tuple"),
    ],
)
def test_other_row_uses_are_unsupported(function: Callable[..., object], reason: str) -> None:
    with pytest.raises(Unsupported, match=reason):
        trace(function, {"image": STR})
