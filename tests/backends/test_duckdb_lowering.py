from collections.abc import Callable

import duckdb
import pytest

from udfx.backends.duckdb import DuckDBBackend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import FLOAT, STR


@pytest.mark.parametrize(
    "function",
    [
        lambda value: value.replace("|", "\x00中'\\").title(),
        lambda value: [part.title() for part in value.split("\x00")],
    ],
)
def test_relational_expression_preserves_nul_literals(
    function: Callable[[str], object],
) -> None:
    backend = DuckDBBackend()
    schema = {"value": STR}
    expr = backend.build(trace(function, schema))
    assert str(expr).startswith("(SELECT ")
    assert "\x00" not in str(expr)
    rows = [(value,) for value in ("", "a|b", "a\x00b", "Σ|ΣA", "ǳǲ|中😀", "'\\\n")]
    assert check_function(function, schema, backend, rows) == []


def test_relational_cast_preserves_faults() -> None:
    def parsed(value: str) -> int:
        try:
            return int(value.replace("\x00", ""))
        except ValueError:
            return -1

    backend = DuckDBBackend()
    schema = {"value": STR}
    expr = backend.build(trace(parsed, schema))
    assert str(expr).startswith("(SELECT ")
    values = ["1\x002", "٣٢", "invalid", "9223372036854775808", "1__2", ""]
    assert check_function(parsed, schema, backend, [(value,) for value in values]) == []


@pytest.mark.parametrize("value", [-0.0, 1.0000000000000002, 0.9999999999999999, 1e-300])
def test_relational_expression_preserves_float_literals(value: float) -> None:
    def selected(text: str) -> float:
        return value if text.title() == "Selected" else 1.0

    backend = DuckDBBackend()
    schema = {"text": STR}
    expr = backend.build(trace(selected, schema))
    assert str(expr).startswith("(SELECT ")
    assert check_function(selected, schema, backend, [("SELECTED",), ("other",)]) == []


def test_relational_expression_keeps_row_order_and_nulls() -> None:
    def title(value: str | None) -> str | None:
        return None if value is None else value.title()

    backend = DuckDBBackend(duckdb.connect(config={"threads": 4}))
    schema = {"value": STR.or_none()}
    rows = [(value,) for value in ("B", None, "a", "", "A", "b", "a", None, "ΣA") * 512]
    expr = backend.build(trace(title, schema))
    assert backend.evaluate([expr], schema, rows) == [[title(value) for (value,) in rows]]


def test_relational_lowering_state_is_local() -> None:
    backend = DuckDBBackend()
    schema = {"value": STR}
    title = backend.build(trace(lambda value: value.title(), schema))
    identity = backend.build(trace(lambda value: value, schema))
    assert str(title).startswith("(SELECT ")
    assert not str(identity).startswith("(SELECT ")


def test_relational_lowering_keeps_distinct_float_inputs() -> None:
    def label(value: float, text: str) -> str:
        return text.title() + str(value)

    backend = DuckDBBackend()
    schema = {"value": FLOAT, "text": STR}
    expr = backend.build(trace(label, schema))
    assert not str(expr).startswith("(SELECT ")
    assert check_function(label, schema, backend, [(0.0, "x"), (-0.0, "x")]) == []


@pytest.mark.parametrize(
    "query, expected",
    [
        ("SELECT value FROM source WHERE {expr} = 'A'", [("a",), ("a",)]),
        ("SELECT {expr}, count(*) FROM source GROUP BY {expr}", [("A", 2), ("B", 1)]),
        ("SELECT max({expr}) FROM source", [("B",)]),
        (
            "SELECT value FROM source LEFT JOIN (VALUES ('A')) AS target(label) ON {expr} = label",
            [("a",), ("a",), ("b",)],
        ),
    ],
)
def test_relational_expression_composes_in_queries(
    query: str, expected: list[tuple[object, ...]]
) -> None:
    backend = DuckDBBackend()
    expr = backend.build(trace(lambda value: value.title(), {"value": STR}))
    source = "WITH source(value) AS (VALUES ('a'), ('b'), ('a')) "
    assert sorted(duckdb.sql(source + query.format(expr=expr)).fetchall()) == expected
