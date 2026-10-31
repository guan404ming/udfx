import json
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params
from pyspark.sql import functions

from udfx.backends.base import Backend
from udfx.backends.spark import json as spark_json
from udfx.frontend import trace
from udfx.ops import OpKey
from udfx.semantics.text import whitespace
from udfx.testing import check_function
from udfx.types import STR, Kind


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"polars", "spark"}]
)
@pytest.mark.parametrize("ensure_ascii", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("copied", [False, True])
def test_json_words_skip_element_encoding(
    backend: Backend[Any],
    ensure_ascii: bool,
    reverse: bool,
    copied: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def packed(text: str) -> str:
        words = text.rsplit() if reverse else text.split()
        return json.dumps([word for word in words] if copied else words, ensure_ascii=ensure_ascii)

    schema = {"text": STR.or_none()}
    rows = [
        (None,),
        ("",),
        (whitespace(),),
        ("  alpha\tbeta\n gamma  ",),
        ('quote" slash\\ zero\x00 delete\x7f',),
        (r"\uABCD \\uFFFF",),
        ("中文 😀 café",),
        *((f"left{character}right",) for character in whitespace()),
    ]
    rules = (
        {
            OpKey(name, (Kind.STR,)): lambda value: functions.raise_error(
                "Unexpected word array"
            ).cast("array<string>")
            for name in (".split", ".rsplit")
        }
        if backend.name == "spark"
        else {}
    )
    if backend.name == "spark":
        monkeypatch.setattr(
            spark_json,
            "ENCODING",
            replace(
                spark_json.ENCODING,
                replace=lambda *arguments: functions.raise_error("Unexpected array padding").cast(
                    "string"
                ),
            ),
        )
    original = backend.rules.get
    with patch.object(backend.rules, "get", lambda key: rules.get(key) or original(key)):
        assert check_function(packed, schema, backend, rows) == []
        lowered = backend.lower_program(trace(packed, schema))
    if backend.name == "spark":
        assert lowered.exact is None
        return
    assert ".list.eval(" not in str(lowered.expr)
    assert lowered.exact is not None
    flags = backend.evaluate([lowered.exact], schema, rows)[0]
    assert flags[1:6] == [True] * 5
    assert flags[6] is (not ensure_ascii)
    assert all(flags[7:])


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"polars", "spark"}]
)
def test_json_word_mapping_keeps_element_encoding(backend: Backend[Any]) -> None:
    def packed(text: str) -> str:
        return json.dumps([word + " value" for word in text.split()])

    schema = {"text": STR}
    assert check_function(packed, schema, backend, [("a b",), ("",), ("é 😀",)]) == []
    if backend.name == "polars":
        assert ".list.eval(" in str(backend.lower_program(trace(packed, schema)).expr)
