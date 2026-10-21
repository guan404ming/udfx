import ast

import pytest

from bench.mine.measurement import measure
from bench.mine.models import Candidate
from bench.mine.schema_search import schemas
from udfx.types import BOOL, DATETIME, FLOAT, INT, STR


def test_schema_search_compiles_mixed_parameter_types() -> None:
    node = ast.parse("lambda text, number: text.lower() + str(number / 2)", mode="eval").body
    assert isinstance(node, ast.Lambda)
    result = measure(Candidate("notebook", 1, node))
    assert result.ok
    assert result.schema == {"text": "str?", "number": "float?"}


def test_schema_search_limits_attempts_and_varies_every_parameter() -> None:
    node = ast.parse("lambda first, second, third, fourth, fifth: first", mode="eval").body
    assert isinstance(node, ast.Lambda)
    attempts = list(schemas(Candidate("notebook", 1, node)))
    assert len(attempts) == 64
    assert len({tuple(schema.values()) for schema in attempts}) == 64
    for name in attempts[0]:
        assert {schema[name] for schema in attempts} == {
            STR.or_none(),
            INT.or_none(),
            FLOAT.or_none(),
            BOOL.or_none(),
        }


def test_schema_search_includes_boolean_type() -> None:
    node = ast.parse("lambda value: not value", mode="eval").body
    assert isinstance(node, ast.Lambda)
    attempts = list(schemas(Candidate("notebook", 1, node)))
    assert len(attempts) == 4
    assert attempts[0] == {"value": BOOL.or_none()}
    assert measure(Candidate("notebook", 1, node)).schema == {"value": "bool?"}


def test_row_columns_use_mixed_hints_and_reach_compiler() -> None:
    node = ast.parse("lambda row: row['text'].lower() + str(row.amount / 2)", mode="eval").body
    assert isinstance(node, ast.Lambda)
    candidate = Candidate("notebook", 1, node, kind="row")
    attempts = list(schemas(candidate))
    assert attempts[0] == {"text": STR.or_none(), "amount": FLOAT.or_none()}
    result = measure(candidate)
    assert result.ok
    assert result.schema == {"text": "str?", "amount": "float?"}


@pytest.mark.parametrize(
    ("source", "category"),
    [
        ("lambda row: row[0]", "row columns could not be inferred"),
        ("lambda row, other: row['value']", "row candidate requires one parameter"),
        ("lambda row: row.row", "row column names match parameter names"),
    ],
)
def test_row_inference_reports_unsupported_shapes(source: str, category: str) -> None:
    node = ast.parse(source, mode="eval").body
    assert isinstance(node, ast.Lambda)
    result = measure(Candidate("notebook", 1, node, kind="row"))
    assert not result.ok
    assert result.category == category


@pytest.mark.parametrize(
    "source",
    [
        "lambda value: value.year",
        "lambda value: value.month",
        "lambda value: value.day",
        "lambda value: value.hour",
        "lambda value: value.minute",
        "lambda value: value.second",
        "lambda value: value.microsecond",
        "lambda value: value.strftime('%Y-%m-%d')",
        "lambda value: value.weekday()",
    ],
)
def test_schema_search_infers_datetime(source: str) -> None:
    node = ast.parse(source, mode="eval").body
    assert isinstance(node, ast.Lambda)
    candidate = Candidate("notebook", 1, node)
    assert next(schemas(candidate)) == {"value": DATETIME.or_none()}
    result = measure(candidate)
    assert result.ok
    assert result.schema == {"value": "datetime?"}


def test_row_schema_search_infers_datetime() -> None:
    node = ast.parse("lambda row: row['time'].year + row.amount", mode="eval").body
    assert isinstance(node, ast.Lambda)
    candidate = Candidate("notebook", 1, node, kind="row")
    assert next(schemas(candidate)) == {"time": DATETIME.or_none(), "amount": FLOAT.or_none()}
    assert measure(candidate).ok
