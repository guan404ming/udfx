import json
from typing import Any

import pytest
from conftest import backend_params
from pyspark.sql import functions

from udfx.backends.base import Backend
from udfx.ir import Col
from udfx.ops import apply
from udfx.semantics.json import simple_array
from udfx.testing.json_traps import JSON_TEXT_TRAPS, JSON_TRAPS
from udfx.types import JSON


@pytest.mark.parametrize(
    "backend", [value for value in backend_params() if value.id in {"spark", "spark-ansi-off"}]
)
def test_total_json_decoder_keeps_its_guard(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = []
    original = functions.from_json

    def decode(value: Any, *arguments: Any, **keywords: Any) -> Any:
        inputs.append(value)
        return original(value, *arguments, **keywords)

    monkeypatch.setattr(functions, "from_json", decode)
    value = Col("value", JSON)
    lowered = backend.lower(apply("json.result.list[str]", value))
    assert len(inputs) == 1
    assert str(inputs[0]) == str(backend.column(value))
    sources = [
        *JSON_TEXT_TRAPS,
        *JSON_TRAPS,
        "[]",
        '["a", "é", "😀"]',
        r'["a\n", "b\\c"]',
        r'["\u0061"]',
        '["a",]',
        '["a"] []',
        "[" * 2000 + "0" + "]" * 2000,
        '["' + "a" * 20000,
    ]
    rows = [(source,) for source in sources]
    values, flags = backend.evaluate(
        [lowered.expr, backend.exact_expression(lowered)], {"value": JSON}, rows
    )
    assert flags == [simple_array(source) for source in sources]
    for source, actual, exact in zip(sources, values, flags, strict=True):
        if exact:
            assert actual == json.loads(source)
