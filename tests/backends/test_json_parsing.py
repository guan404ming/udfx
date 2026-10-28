import json
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import Call, Col, children, lit
from udfx.ops import OpKey, apply
from udfx.semantics.assumptions import assuming
from udfx.testing import check_function
from udfx.types import STR, Kind

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
    json.dumps(["ab" * 10000 + "\n" + "cd" * 10000]),
    '["' + "a" * 20000 + r'\q"]',
    '["' + "a" * 20000,
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


def test_json_sort_serializes_once() -> None:
    program = trace(sorted_json, {"text": STR})
    seen = set()
    pending = [program.value]
    serialized = []
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Call) and node.op.name == "json.dumps":
            serialized.append(node)
        pending.extend(children(node))
    assert len(serialized) == 1
    assert serialized[0].op.key == OpKey("json.dumps", (Kind.LIST, Kind.BOOL))
    source = serialized[0].arguments[0]
    assert isinstance(source, Call) and source.op.name == "sorted"


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("name", ["json.item", "json.length"])
def test_validated_json_array_reuses_native_value(backend: Backend[Any], name: str) -> None:
    if backend.name not in {"duckdb", "spark"}:
        pytest.skip("Native JSON containers")
    text = Col("text", STR)
    value = apply("json.value", text)
    node = apply(name, value, lit(0)) if name == "json.item" else apply(name, value)
    with (
        assuming(apply("json.valid", text), True),
        assuming(apply("json.is_type", text, lit("list")), True),
    ):
        lowered = backend.lower(node)
    assert backend.exact(lowered) is None


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
def test_json_invalid_prefix_uses_native_exception_branch(backend: Backend[Any]) -> None:
    schema = {"text": STR}
    rows = [(value,) for value in ("", " \t\r\n", "ordinary text", "\ufeff[]", "+1", ".2", "-x")]
    program = trace(counted, schema)
    lowered = backend.lower_program(program)
    values, exact = backend.evaluate(
        [lowered.expr, backend.exact_expression(lowered)], schema, rows
    )
    assert values == [-1] * len(rows)
    assert exact == [True] * len(rows)


@pytest.mark.parametrize("backend", backend_params())
def test_json_token_type_skips_container_conversion(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def distinct(text: str) -> int:
        try:
            values = [tuple(value) if type(value) is list else value for value in json.loads(text)]
            return len(set(values))
        except (ValueError, TypeError):
            return -1

    def forbidden(*args: object) -> Any:
        pytest.fail("Converted a JSON string token as a container")

    original = backend.rules.get
    conversion = OpKey("json.dumps", (Kind.STR, Kind.BOOL))
    monkeypatch.setattr(
        backend.rules, "get", lambda key: forbidden if key == conversion else original(key)
    )
    schema = {"text": STR.or_none()}
    assert check_function(distinct, schema, backend, [(value,) for value in JSON_VALUES]) == []
    lowered = backend.lower_program(trace(distinct, schema))
    rows = [('["a", "b"]',), ('[["a"], ["b"]]',), ('[1, "two", null]',)]
    assert backend.evaluate([backend.exact_expression(lowered)], schema, rows) == [
        [True, False, False]
    ]


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
