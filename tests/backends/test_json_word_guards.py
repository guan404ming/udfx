import json
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.semantics.text import whitespace
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "datafusion"}]
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("copied", [False, True])
def test_json_word_guard_uses_source(backend: Backend[Any], reverse: bool, copied: bool) -> None:
    def packed(text: str) -> str:
        words = text.rsplit() if reverse else text.split()
        return json.dumps([word for word in words] if copied else words)

    schema = {"text": STR.or_none()}
    rows = [
        (None,),
        ("",),
        (whitespace(),),
        ("  alpha\tbeta\n gamma  ",),
        ('quote" slash\\ zero\x00 delete\x7f',),
        (r"\uABCD \\uFFFF",),
        ("中文 😀 café",),
        ("a\x80b",),
        *((f"left{character}right",) for character in whitespace()),
    ]
    assert check_function(packed, schema, backend, rows) == []
    lowered = backend.lower_program(trace(packed, schema))
    assert lowered.exact is not None
    assert "list_transform" not in str(lowered.exact).lower()
    assert "array_transform" not in str(lowered.exact).lower()
    flags = backend.evaluate([lowered.exact], schema, rows)[0]
    assert flags[1:6] == [True] * 5
    assert flags[6:8] == [False, False]
    assert all(flags[8:])


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "datafusion"}]
)
def test_json_word_mapping_checks_mapped_values(backend: Backend[Any]) -> None:
    def packed(text: str) -> str:
        return json.dumps([word + "é" for word in text.split()])

    schema = {"text": STR}
    rows = [("a b",), ("",), ("é 😀",)]
    assert check_function(packed, schema, backend, rows) == []
    lowered = backend.lower_program(trace(packed, schema))
    assert lowered.exact is not None
    assert backend.evaluate([lowered.exact], schema, rows)[0] == [False, True, False]
