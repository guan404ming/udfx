import json
from typing import Any

import pytest
from conftest import backend_params

from udfx import compile
from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Col, lit
from udfx.ops import apply
from udfx.types import INT, JSON, STR


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "spark"}]
)
@pytest.mark.parametrize("indexes", [(0,), (0, 0), (0, 1)])
def test_json_indexed_object_check_skips_serialization(
    backend: Backend[Any], indexes: tuple[int, ...]
) -> None:
    value = Col("text", JSON)
    for index in indexes:
        value = apply("json.item", value, lit(index))
    predicate = apply("json.get.AttributeError", value, lit("value"), lit("null", JSON))
    expression = str(backend.lower(predicate).expr).lower()
    if backend.name == "spark":
        assert ("struct<>" in expression) is any(indexes)
        assert ("try_parse_json" in expression) is any(indexes)
        assert "map<" not in expression
        assert "get_json_object" not in expression
    else:
        assert "json_type" in expression
        assert "".join(f"/{index}" for index in indexes) in expression
        assert "json_extract" not in expression


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "spark"}]
)
@pytest.mark.parametrize("position", [None, 0, 1])
def test_json_indexed_projection_preserves_faults(
    backend: Backend[Any], position: int | None
) -> None:
    def selected(text: str, key: str | None, default: str | None) -> object:
        try:
            value = json.loads(text)[0]
            if position is not None:
                value = value[position]
            return value.get(key, default)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            return "invalid"

    schema = {"text": STR, "key": STR.or_none(), "default": STR.or_none()}
    program = trace(selected, schema, result=STR.or_none())
    expression = str(backend.lower_program(program).expr)
    prefix = f"$[0][{position}]" if position is not None else "$[0]"
    if backend.name == "duckdb":
        prefix = f"/0/{position}" if position is not None else "/0"
    assert prefix in expression
    documents = [
        "{}",
        '{"value":"found"}',
        '{"value":null}',
        '{"value":"first","value":"last"}',
        '{"nested":{"value":"other"},"value":"found"}',
        '{"nested":[{},true,null,["text",1]],"value":"found"}',
        '{"":"empty","~a/b":"escaped","a.b":"dot","x[y]":"brackets"}',
        '{"a\'b":"quote","*":"star","中文":"unicode"}',
        '"text"',
        "[]",
        "null",
        "1",
    ]
    rows = [
        (
            f"[[{'null,' if position == 1 else ''}{document}]]"
            if position is not None
            else f"[{document}]",
            key,
            default,
        )
        for document in documents
        for key in ("value", "", "~a/b", "a.b", "x[y]", "a'b", "*", "中文", None)
        for default in (None, "missing")
    ]
    rows.extend((text, "value", None) for text in ("broken", "[]", "[[]]", "{}", "null"))
    translation = compile(selected, schema, backend, result=STR.or_none())
    assert translation.ok, translation.unsupported
    assert backend.evaluate([translation.expr], schema, rows) == [[selected(*row) for row in rows]]
    wrong = (
        f'[[{"null," if position == 1 else ""}{{"value":1}}]]'
        if position is not None
        else '[{"value":1}]'
    )
    with pytest.raises(OverflowError, match="does not fit"):
        backend.evaluate([translation.expr], schema, [(wrong, "value", None)])


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "spark"}]
)
def test_json_dynamic_index_keeps_python_bounds(backend: Backend[Any]) -> None:
    def selected(text: str, position: int) -> object:
        try:
            return json.loads(text)[position].get("value")
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            return "invalid"

    schema = {"text": STR, "position": INT}
    translation = compile(selected, schema, backend, result=STR.or_none())
    assert translation.ok, translation.unsupported
    rows = [('[{"value":"first"},{"value":"last"}]', index) for index in (-3, -1, 0, 1, 2)]
    assert backend.evaluate([translation.expr], schema, rows) == [[selected(*row) for row in rows]]


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "spark"}]
)
@pytest.mark.parametrize("position", [-1, 2**31, 2**63 - 1])
def test_json_literal_index_keeps_python_bounds(backend: Backend[Any], position: int) -> None:
    def selected(text: str) -> object:
        try:
            return json.loads(text)[position].get("value")
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            return "invalid"

    schema = {"text": STR}
    translation = compile(selected, schema, backend, result=STR.or_none())
    assert translation.ok, translation.unsupported
    rows = [(text,) for text in ('[{"value":"found"}]', "[]", "{}", "null")]
    assert backend.evaluate([translation.expr], schema, rows) == [[selected(*row) for row in rows]]


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"spark", "spark-ansi-off"}]
)
@pytest.mark.parametrize("indexed", [False, True])
def test_json_variant_keys_stay_native(backend: Backend[Any], indexed: bool) -> None:
    def selected(text: str, key: str | None, default: str | None) -> object:
        value = json.loads(text)
        if indexed:
            value = value[0]
        return value.get(key, default)

    schema = {"text": STR, "key": STR.or_none(), "default": STR.or_none()}
    program = trace(selected, schema, result=STR.or_none())
    lowered = backend.lower_program(program)
    assert lowered.exact is not None
    keys = ("", "a\\b", "*", "\x00", "\t", "\n", 'a"b', "a'b", "a'\"b", "x[y]", "a.b", "漢")
    rows = [
        (
            json.dumps([document] if indexed else document),
            key,
            default,
        )
        for key in (*keys, None)
        for document in ({name: "found" for name in keys}, {}, {name: None for name in keys})
        for default in (None, "missing")
    ]
    actual, native = backend.evaluate([lowered.expr, lowered.exact], schema, rows)
    assert actual == program.run(rows)
    assert native == [True] * len(rows)
