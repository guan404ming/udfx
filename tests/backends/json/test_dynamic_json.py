import json
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx import compile
from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import BOOL, DATETIME, FLOAT, INT, STR, Type, list_of

DOCUMENTS = [
    "null",
    "true",
    "false",
    "0",
    "1.25",
    "NaN",
    "Infinity",
    "-Infinity",
    "[]",
    "{}",
    '"hello"',
    '{"value": null}',
    '{"value": "name"}',
    '{"value": true}',
    '{"value": [1, "two", null, {"key": 3}]}',
    '{"value": {"key": ["nested", 9007199254740993]}}',
    '{"value": 1234567890123456789012345678901234567890}',
    '{"value": NaN}',
    '{"value": -0.0}',
    '{"value": 1e999}',
    '{"value": "first", "value": "last"}',
    r'{"value": "\ud83d\ude00"}',
    r'{"value": "\ud800"}',
    '[{"value": "first"}, {"value": "second"}]',
    "broken",
    "0 ൎ",
    "true false",
    "null null",
    "{} []",
    "{}]garbage[{}",
    '{"value":"first"} {"value":"second"}',
    '{"value":"first"} trailing',
    '{"value":"first","nested":' + "[" * 600 + "0" + "]" * 600 + "}",
    '{"value":"first","large":' + "9" * 4500 + "}",
    "[1,]",
    '"unterminated',
]


def selected(text: str, key: str) -> str:
    try:
        return json.dumps(json.loads(text).get(key), ensure_ascii=True)
    except (TypeError, ValueError, AttributeError):
        return "invalid"


def nested(text: str, key: str) -> str:
    try:
        return json.dumps(json.loads(text)[key]["key"][0])
    except (TypeError, ValueError, AttributeError, KeyError, IndexError):
        return "invalid"


def converted(text: str, key: str) -> str:
    try:
        return str(int(json.loads(text)[key]))
    except (TypeError, ValueError, AttributeError, KeyError, OverflowError):
        return "invalid"


def classified(text: str, key: str) -> str:
    try:
        value = json.loads(text).get(key)
        if value is None:
            return "null"
        if type(value) is dict:
            return "dict"
        if type(value) is list:
            return "list"
        return str(bool(value))
    except (TypeError, ValueError, AttributeError):
        return "invalid"


def extracted(text: str, key: str) -> object:
    try:
        value = json.loads(text)
        if isinstance(value, list):
            for item in value:
                return item.get(key)
        elif isinstance(value, dict):
            return value.get(key)
        return None
    except (TypeError, ValueError, AttributeError):
        return None


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [selected, nested, converted, classified])
def test_dynamic_json_semantics(backend: Backend[Any], function: Callable[..., object]) -> None:
    rows = [(text, "value") for text in DOCUMENTS]
    assert check_function(function, {"text": STR, "key": STR}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_dynamic_json_result_contract(backend: Backend[Any]) -> None:
    schema = {"text": STR, "key": STR}
    translation = compile(extracted, schema, backend, result=STR.or_none())
    assert translation.ok, translation.unsupported
    rows = [
        (value, "value")
        for value in [
            '{"value": "x"}',
            '{"other": 1}',
            '[{"value": "first"}, {"value": "second"}]',
            "[]",
            "{}",
            "null",
            "broken",
            "[1]",
            '[{"value": null}]',
        ]
    ]
    assert backend.evaluate([translation.expr], schema, rows) == [[extracted(*row) for row in rows]]
    with pytest.raises(OverflowError, match="does not fit"):
        backend.evaluate([translation.expr], schema, [('{"value": 123}', "value")])


def test_dynamic_json_return_requires_contract() -> None:
    translation = compile(extracted, {"text": STR, "key": STR})
    assert not translation.ok
    assert "result type" in str(translation.unsupported)


@pytest.mark.parametrize("other", [DATETIME, list_of(DATETIME), list_of(list_of(STR))])
def test_json_merge_rejects_unrepresented_values(other: Type) -> None:
    translation = compile(
        lambda text, value: json.loads(text) if text else value,
        {"text": STR, "value": other},
        result=STR,
    )
    assert not translation.ok
    assert "return types differ" in str(translation.unsupported)


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("result", "value", "invalid"),
    [
        (INT, 9007199254740993, 2**63),
        (BOOL, True, 1),
        (FLOAT, 1.25, "1.25"),
        (list_of(STR.or_none()), ["a", None], ["a", 1]),
    ],
)
def test_json_typed_outputs(
    backend: Backend[Any], result: Type, value: object, invalid: object
) -> None:
    schema = {"text": STR, "key": STR}
    translation = compile(extracted, schema, backend, result=result)
    assert translation.ok, translation.unsupported
    output = backend.evaluate([translation.expr], schema, [(json.dumps({"value": value}), "value")])
    assert output == [[value]]
    with pytest.raises(OverflowError, match="does not fit"):
        backend.evaluate([translation.expr], schema, [(json.dumps({"value": invalid}), "value")])
    with pytest.raises(OverflowError, match="does not fit"):
        backend.evaluate([translation.expr], schema, [("{}", "value")])


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in ("duckdb", "spark")]
)
def test_dynamic_json_native_extraction(backend: Backend[Any]) -> None:
    schema = {"text": STR, "key": STR}
    rows = [
        ('{"nested":{"a":[1,true,false,null]},"value":"first"}', "value"),
        ('[{"nested":{"a":[1,true,false,null]},"value":"first"},{"value":"second"}]', "value"),
        ('{"other":"different"}', "other"),
        ('{"value":null}', "value"),
        ('{"other":1}', "value"),
        ("[]", "value"),
        ("[1]", "value"),
        ("null", "value"),
        ('{"value":"中文😀"}', "value"),
        (r'{"value":"\ud83d\ude00"}', "value"),
        ('{"a.b/~[x]":"path"}', "a.b/~[x]"),
    ]
    program = trace(extracted, schema, result=STR.or_none())
    lowered = backend.lower_program(program)
    values, exact = backend.evaluate(
        [lowered.expr, backend.exact_expression(lowered)], schema, rows
    )
    assert values == [extracted(*row) for row in rows]
    assert exact == [True] * len(rows)


@pytest.mark.parametrize("backend", backend_params())
def test_dynamic_json_keys_preserve_python_lookup(backend: Backend[Any]) -> None:
    schema = {"text": STR, "key": STR}
    keys = ["", "*", "a'b", 'a"b', "a\\b", "]", ".", "中文", "\n", "\x00"]
    rows = [(json.dumps({key: "found", "other": "wrong"}), key) for key in keys]
    translation = compile(extracted, schema, backend, result=STR.or_none())
    assert translation.ok, translation.unsupported
    assert backend.evaluate([translation.expr], schema, rows) == [["found"] * len(rows)]


def nested_value(text: str, key: str) -> object:
    try:
        return json.loads(text)[key]["items"][1]
    except (ValueError, TypeError, KeyError, IndexError):
        return None


@pytest.mark.parametrize("backend", backend_params())
def test_native_json_nested_indexing(backend: Backend[Any]) -> None:
    schema = {"text": STR, "key": STR}
    rows = [
        ('{"outer":{"items":[null,"found"]}}', "outer"),
        ('{"other":{"items":["first","found","last"]}}', "other"),
        ('{"outer":{"items":[]}}', "outer"),
        ('{"outer":{"items":["first"]}}', "outer"),
        ('{"outer":{}}', "outer"),
        ("{}", "outer"),
        ('{"outer":{"items":"abc"}}', "outer"),
    ]
    program = trace(nested_value, schema, result=STR.or_none())
    translation = compile(nested_value, schema, backend, result=STR.or_none())
    assert translation.ok, translation.unsupported
    assert backend.evaluate([translation.expr], schema, rows) == [
        [nested_value(*row) for row in rows]
    ]
    if backend.name in {"duckdb", "spark"}:
        lowered = backend.lower_program(program)
        assert backend.evaluate([backend.exact_expression(lowered)], schema, rows[:6]) == [
            [True] * 6
        ]
