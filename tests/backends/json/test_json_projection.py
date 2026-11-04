import json
from typing import Any

import pytest
from conftest import backend_params

from udfx import compile
from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.types import STR


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "spark"}]
)
@pytest.mark.parametrize("nullable", [False, True])
def test_json_string_projection_preserves_defaults(backend: Backend[Any], nullable: bool) -> None:
    def selected(text: str, key: str | None, default: str | None) -> object:
        try:
            return json.loads(text).get(key, default)
        except (ValueError, AttributeError):
            return "invalid"

    schema = {"text": STR, "key": STR.or_none(), "default": STR.or_none()}
    result = STR.or_none() if nullable else STR
    program = trace(selected, schema, result=result)
    expression = str(backend.lower_program(program).expr)
    assert ("variant_get(" if backend.name == "spark" else "json_type(") in expression
    assert "$[*]" not in expression
    rows = [
        ('{"value":"found"}', "value", None),
        ('{"value":"found"}', "value", "unused"),
        ('{"value":"found"}', None, "missing"),
        ('{"":"ignored"}', None, "missing"),
        ("{}", "value", '"\\\n中文'),
        ("[]", "value", "unused"),
        ("broken", "value", None),
    ]
    if nullable:
        rows.extend([('{"value":null}', "value", "unused"), ("{}", "value", None)])
    translation = compile(selected, schema, backend, result=result)
    assert translation.ok, translation.unsupported
    assert backend.evaluate([translation.expr], schema, rows) == [[selected(*row) for row in rows]]
    with pytest.raises(OverflowError, match="does not fit"):
        backend.evaluate([translation.expr], schema, [('{"value":1}', "value", "unused")])
