import json
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR

JSON_VALUES = [
    None,
    "",
    "[]",
    '["b a", "c", "b a", ""]',
    ' \n[ "a", "b" ]\t',
    '["中文", "😀", "é"]',
    json.dumps(['a"b', "a\\b", "\b\f\n\r\t/", "\\n"]),
    r'["\u0061", "\ud83d\ude00", "\u0000"]',
    '"ab c"',
    '[1, "two", null]',
    '[["a"], ["b"]]',
    '{"a": "x", "b": 1}',
    "null",
    "true",
    "123",
    "NaN",
    '["a",]',
    r'["bad\escape"]',
    '["a\x00b"]',
    '["a"] trailing',
    '["' + "x" * 20000 + '"]',
    json.dumps(["\\n" * 2000]),
    json.dumps(["a"] * 2000),
]


def counted(text: str) -> int:
    try:
        return int(len(json.loads(text)))  # noqa: RUF046
    except (ValueError, TypeError):
        return -1


def sorted_json(text: str) -> str:
    try:
        return json.dumps(sorted(json.loads(text)), ensure_ascii=False)
    except (ValueError, TypeError):
        return "invalid"


def cleaned(text: str) -> str:
    try:
        return json.dumps([word.lower() for word in json.loads(text)], ensure_ascii=False)
    except (ValueError, TypeError, AttributeError):
        return "invalid"


def occurrences(text: str) -> int:
    try:
        return json.loads(text).count("a")
    except (ValueError, TypeError, AttributeError):
        return -1


def cardinality(text: str) -> int:
    try:
        return len(set(json.loads(text)))
    except (ValueError, TypeError):
        return -1


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [counted, sorted_json, cleaned, occurrences, cardinality])
def test_json_consumers(function: Callable[..., object], backend: Backend[Any]) -> None:
    assert (
        check_function(function, {"text": STR.or_none()}, backend, [(x,) for x in JSON_VALUES])
        == []
    )


@pytest.mark.parametrize("backend", backend_params())
def test_json_parser_native_domain(backend: Backend[Any]) -> None:
    schema = {"text": STR}
    program = trace(counted, schema)
    lowered = backend.lower_program(program)
    rows = [(value,) for value in JSON_VALUES[2:8]]
    flags = backend.evaluate([backend.exact_expression(lowered)], schema, rows)[0]
    assert flags == [True, True, True, True, True, False]


@pytest.mark.parametrize("backend", backend_params())
def test_json_uncaught_errors(backend: Backend[Any]) -> None:
    def count(text: str) -> int:
        return len(json.loads(text))

    assert check_function(count, {"text": STR}, backend, [("[]",), ('["a"]',), ("broken",)]) == []
    expr = backend.build(trace(count, {"text": STR}))
    text = '["a",\n invalid]'
    with pytest.raises(json.JSONDecodeError) as raised:
        backend.evaluate([expr], {"text": STR}, [(text,)])
    assert raised.value.doc == text
    assert raised.value.pos == 7


@pytest.mark.parametrize("backend", backend_params())
def test_json_type_branches(backend: Backend[Any]) -> None:
    def classified(text: str) -> str:
        try:
            value = json.loads(text)
            if type(value) is list:
                return "list"
            if isinstance(value, str):
                return "str"
            return "other"
        except ValueError:
            return "invalid"

    rows = [(text,) for text in ("[]", '"a"', "true", "123", "null", "{}", "bad")]
    assert check_function(classified, {"text": STR}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "source",
    ['["a"]', "not json", "9" * 5000, "[" * 2000 + "]" * 2000],
    ids=["valid", "syntax", "integer-limit", "depth-limit"],
)
def test_json_constant_errors(backend: Backend[Any], source: str) -> None:
    def classified(text: str) -> str:
        try:
            json.loads(source)
            return text
        except json.JSONDecodeError:
            return "syntax"
        except ValueError:
            return "value"
        except RecursionError:
            return "depth"

    assert check_function(classified, {"text": STR}, backend, [("valid",)]) == []


@pytest.mark.parametrize(
    "function",
    [
        lambda text: json.loads(s=text),
        lambda text: sorted(json.loads(text)),
        lambda text: list(json.loads(text)),
        lambda text: tuple(json.loads(text)),
        lambda text: sorted(set(json.loads(text))),
        lambda text: [word for word in json.loads(text)],
        lambda text: json.loads(text, strict=False),
        lambda text: json.loads(text, parse_int=str),
    ],
)
def test_dynamic_json_result_stays_unsupported(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported):
        trace(function, {"text": STR})
