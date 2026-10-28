import json
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.ir import Col, If, lit
from udfx.ops import OpKey, apply
from udfx.semantics.assumptions import assuming
from udfx.testing import check_function
from udfx.types import STR, Kind


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"polars", "datafusion"}]
)
@pytest.mark.parametrize("known", [False, True])
def test_json_decoder_reuses_string_validation(backend: Backend[Any], known: bool) -> None:
    column = Col("value", STR)
    validated = apply("json.string_native", column)
    value = apply("json.result.str", apply("json.value", column))
    with assuming(validated, known):
        lowered = backend.lower(value)
    expression = str(lowered.expr).lower()
    conditional = "case when" if backend.name == "datafusion" else ".when("
    assert (conditional not in expression) is known
    guarded = If(validated, value, lit("invalid"), STR)
    expression = backend.lower(guarded).expr
    rows = [('"line\\n"',), ("invalid",), ('"unfinished',), ("null",), ("[]",)]
    assert backend.evaluate([expression], {"value": STR}, rows) == [
        ["line\n", "invalid", "invalid", "invalid", "invalid"]
    ]


@pytest.mark.parametrize("backend", backend_params())
def test_json_array_prefix_removes_string_branch(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def counted(text: str) -> int:
        try:
            return len(json.loads(text)) if text[0] == "[" else 0
        except (ValueError, IndexError, TypeError):
            return -1

    def forbidden(*args: object) -> Any:
        pytest.fail("Decoded a JSON string after an array prefix")

    original = backend.rules.get
    conversion = OpKey("json.result.str", (Kind.JSON,))
    monkeypatch.setattr(
        backend.rules, "get", lambda key: forbidden if key == conversion else original(key)
    )
    values = [None, "", "[]", '["a"]', '[null,{"x":[1,2]}]', '["', '["a",]', '"text"']
    assert check_function(counted, {"text": STR.or_none()}, backend, [(x,) for x in values]) == []
