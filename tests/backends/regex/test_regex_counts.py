import json
import re
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Col
from udfx.ops import OpKey, apply
from udfx.testing import check_function
from udfx.types import STR, Kind


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("aggregate", [len, bool])
@pytest.mark.parametrize("pattern", [r"([a-z]+)|\d+", r".*?"])
def test_regex_cardinality_does_not_build_match_arrays(
    backend: Backend[Any],
    aggregate: Callable[..., object],
    pattern: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def counted(text: str) -> object:
        return aggregate(re.findall(pattern, text))

    def forbidden(*arguments: object) -> Any:
        pytest.fail("Built an unused match array")

    original = backend.rules.get
    matches = OpKey("re.findall", (Kind.STR, Kind.STR, Kind.INT))
    monkeypatch.setattr(
        backend.rules, "get", lambda key: forbidden if key == matches else original(key)
    )
    schema = {"text": STR.or_none()}
    rows = [(value,) for value in (None, "", "abc12 def3", "é\n42", "a\x00b")]
    assert check_function(counted, schema, backend, rows) == []
    lowered = backend.lower_program(trace(counted, schema))
    if pattern != r".*?" or aggregate is bool:
        assert backend.evaluate([backend.exact_expression(lowered)], schema, rows[1:]) == [
            [True] * (len(rows) - 1)
        ]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("aggregate", [len, bool])
def test_json_cardinality_does_not_build_token_arrays(
    backend: Backend[Any],
    aggregate: Callable[..., object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*arguments: object) -> Any:
        pytest.fail("Built an unused JSON token array")

    original = backend.rules.get
    tokens = OpKey("json.tokens", (Kind.STR,))
    monkeypatch.setattr(
        backend.rules, "get", lambda key: forbidden if key == tokens else original(key)
    )
    source = Col("text", STR)
    lowered = backend.lower(apply(aggregate.__name__, apply("json.tokens", source)))
    values = ["[]", '["", ""]', '["a\\n", "b\\\\", "中文"]', '["a",]', "[1]", '"a"']
    outputs, exact = backend.evaluate(
        [lowered.expr, backend.exact_expression(lowered)], {"text": STR}, [(x,) for x in values]
    )
    assert outputs[:3] == [aggregate(json.loads(value)) for value in values[:3]]
    assert exact == [True, True, True, False, False, False]
