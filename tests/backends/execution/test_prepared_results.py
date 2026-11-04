import json
from dataclasses import replace
from typing import Any

import pytest
from conftest import backend_params
from pyspark.sql import functions

from udfx.backends.base import Backend
from udfx.backends.rules import Lowered
from udfx.frontend import trace
from udfx.ops import OpKey
from udfx.types import BOOL, STR, Kind


def selected(text: str, key: str, enabled: bool) -> object:
    if not enabled:
        return "disabled"
    try:
        return json.loads(text).get(key)
    except (ValueError, AttributeError):
        return None


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"spark", "spark-ansi-off"}]
)
def test_json_result_consumes_prepared_field(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    key = OpKey("json.get.str?", (Kind.JSON, Kind.STR, Kind.JSON))
    original = backend.rules.get(key)
    assert original is not None

    def unused(*arguments: Any, **facts: Any) -> Lowered[Any]:
        value = original(*arguments, **facts)
        assert isinstance(value, Lowered)
        return replace(value, expr=functions.raise_error("unshared projection").cast("string"))

    monkeypatch.setitem(backend.rules._rules, key, unused)  # pyright: ignore[reportPrivateUsage]
    program = trace(selected, {"text": STR, "key": STR, "enabled": BOOL}, result=STR.or_none())
    expression = backend.build(program)
    rows = [
        (text, key, enabled)
        for text in ('{"value":"a"}', '{"value":null}', "{}", "[]", "bad", '{"a\'b":"b"}')
        for key in ("value", "a'b")
        for enabled in (False, True)
    ]
    assert backend.evaluate([expression], program.schema, rows) == [program.run(rows)]
    with pytest.raises(OverflowError, match="does not fit"):
        backend.evaluate([expression], program.schema, [('{"value":1}', "value", True)])


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"spark", "spark-ansi-off"}]
)
@pytest.mark.parametrize("indexes", [(), (0,), (0, 1)])
def test_json_projection_proves_complete_source(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch, indexes: tuple[int, ...]
) -> None:
    def selected(text: str, key: str) -> object:
        try:
            value = json.loads(text)
            for index in indexes:
                value = value[index]
            return value.get(key)
        except (ValueError, IndexError, KeyError, TypeError, AttributeError):
            return "invalid"

    key = OpKey("json.valid", (Kind.STR,))
    original = backend.rules.get(key)
    assert original is not None

    def guarded(value: Any) -> Lowered[Any]:
        lowered = original(value)
        assert isinstance(lowered, Lowered)
        return replace(lowered, exact=functions.lit(False))

    monkeypatch.setitem(backend.rules._rules, key, guarded)  # pyright: ignore[reportPrivateUsage]
    program = trace(selected, {"text": STR, "key": STR}, result=STR.or_none())
    lowered = backend.lower_program(program)
    texts = []
    for document in ('{"value":"found"}', '{"value":null}', "{}"):
        text = document
        for index in reversed(indexes):
            text = "[" + "null," * index + text + "]"
        texts.append(text)
    rows = [(text, "value") for text in (*texts, texts[0] + " trailing", texts[0] + "[]")]
    native = backend.evaluate([backend.exact_expression(lowered)], program.schema, rows)[0]
    assert native[:3] == [True, True, True]
    expression = backend.compose(lowered, program)
    assert backend.evaluate([expression], program.schema, rows) == [program.run(rows)]
