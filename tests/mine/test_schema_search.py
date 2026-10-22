import ast
from pathlib import Path

import pytest

from bench.mine.bindings import HELPER_DEPTH, notebook_bindings
from bench.mine.measurement import measure
from bench.mine.models import Candidate
from bench.mine.schema_search import schema_columns, schemas
from udfx.types import BOOL, DATETIME, FLOAT, INT, STR

from .helpers import collect_sources


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
    ("source", "expected"),
    [
        ("lambda row: row[0] + ':' + str(row[1] / 2)", {"c0": "str?", "c1": "float?"}),
        ("lambda row: row[2].year", {"c0": "str?", "c1": "str?", "c2": "datetime?"}),
        ("lambda row: row[-2].lower() + str(row[-1] / 2)", {"c0": "str?", "c1": "float?"}),
    ],
)
def test_row_positions_infer_schema(source: str, expected: dict[str, str]) -> None:
    node = ast.parse(source, mode="eval").body
    assert isinstance(node, ast.Lambda)
    candidate = Candidate("notebook", 1, node, kind="row")
    attempts = list(schemas(candidate))
    assert {name: str(value) for name, value in attempts[0].items()} == expected
    result = measure(candidate)
    assert result.ok
    assert result.schema == expected


@pytest.mark.parametrize(
    ("source", "category"),
    [
        ("lambda row: row[index]", "row columns could not be inferred"),
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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "def get_day_str(record): return record['day'].lower()\n"
            "frame.apply(lambda row: get_day_str(row), axis=1)",
            {"day": STR.or_none()},
        ),
        (
            "def make_path(record): return filename(record)\n"
            "def filename(entry): return entry.image.lower() + str(entry.number / 2)\n"
            "frame.apply(lambda row: make_path(row), axis=1)",
            {"image": STR.or_none(), "number": FLOAT.or_none()},
        ),
        (
            "def helper(prefix, /, *, record): return prefix + record[0].strftime('%Y')\n"
            "frame.apply(lambda row: helper('', record=row), axis=1)",
            {"c0": DATETIME.or_none()},
        ),
        (
            "def helper(record):\n"
            "    feature = 'stays_in_week_nights'\n"
            "    return record[feature] / 2\n"
            "frame.apply(lambda row: helper(row), axis=1)",
            {"stays_in_week_nights": FLOAT.or_none()},
        ),
        (
            "def helper(record):\n"
            "    position: int = -2\n"
            "    return record[position].lower() + str(record[-1] / 2)\n"
            "frame.apply(helper, axis=1)",
            {"c0": STR.or_none(), "c1": FLOAT.or_none()},
        ),
        (
            "def candidate(row):\n"
            "    feature = 'outer'\n"
            "    def helper(record):\n"
            "        feature = 'inner'\n"
            "        return record[feature].lower()\n"
            "    return row[feature].lower() + helper(row)\n"
            "frame.apply(candidate, axis=1)",
            {"outer": STR.or_none(), "inner": STR.or_none()},
        ),
        (
            "def candidate(row):\n"
            "    def helper(): return row.image.lower()\n"
            "    return helper()\n"
            "frame.apply(candidate, axis=1)",
            {"image": STR.or_none()},
        ),
    ],
)
def test_row_helpers_and_local_keys_infer_schema(
    tmp_path: Path, source: str, expected: dict[str, object]
) -> None:
    candidates, _ = collect_sources(tmp_path, [("notebook", source)])
    candidate = candidates[0]
    assert next(schemas(candidate)) == expected
    result = measure(candidate)
    assert result.ok, result.reason
    assert result.schema == {name: str(value) for name, value in expected.items()}


@pytest.mark.parametrize(
    "body",
    [
        "feature = 'image'\n    feature = unknown\n    return row[feature]",
        "def helper():\n        feature = 'image'\n    return row[feature]",
        "return missing(row)",
    ],
)
def test_row_inference_does_not_guess_unresolved_keys_or_helpers(body: str) -> None:
    node = ast.parse(f"def candidate(row):\n    {body}").body[0]
    assert isinstance(node, ast.FunctionDef)
    result = measure(Candidate("notebook", 1, node, kind="row"))
    assert result.category == "row columns could not be inferred"


def test_row_helper_inference_limits_depth_and_stops_cycles() -> None:
    source = "\n".join(
        f"def helper{index}(record): return record.column{index} + helper{index + 1}(record)"
        for index in range(HELPER_DEPTH + 2)
    )
    tree = ast.parse(source + "\ndef recursive(record): return record.value + recursive(record)")
    bindings = notebook_bindings([tree])
    for expression, expected in [
        ("helper0(row)", [f"column{index}" for index in range(HELPER_DEPTH + 1)]),
        ("recursive(row)", ["value"]),
    ]:
        node = ast.parse(f"lambda row: {expression}", mode="eval").body
        assert isinstance(node, ast.Lambda)
        candidate = Candidate("notebook", 1, node, kind="row", bindings=bindings)
        assert schema_columns(candidate) == (expected, "row")


def test_row_inference_does_not_resolve_shadowed_helpers(tmp_path: Path) -> None:
    candidates, _ = collect_sources(
        tmp_path,
        [
            (
                "notebook",
                "def helper(record): return record.image\n"
                "def candidate(row):\n"
                "    helper = unknown\n"
                "    return helper(row)\n"
                "frame.apply(candidate, axis=1)",
            )
        ],
    )
    assert measure(candidates[0]).category == "row columns could not be inferred"
