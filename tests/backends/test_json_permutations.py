import json
from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.frontend.core.json import serialized_strings
from udfx.ir import Call, Node
from udfx.ops import OpKey
from udfx.testing import check_function
from udfx.types import STR, Kind


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "datafusion"}]
)
@pytest.mark.parametrize("words", [False, True])
def test_json_permutation_guard_uses_source(backend: Backend[Any], words: bool) -> None:
    def packed(text: str) -> str:
        try:
            values = json.loads(text)
            if words:
                values = [" ".join(sorted(value.split(" "))) for value in values]
            else:
                values = sorted(list(values))
            return json.dumps(values)
        except (ValueError, TypeError):
            return "invalid"

    schema = {"text": STR.or_none()}
    rows = [
        (None,),
        ("[]",),
        ('["z a", "b  c", ""]',),
        (json.dumps(['quote"', "slash\\", "\b\t\n\f\r", "\x7f"], ensure_ascii=False),),
        (json.dumps([r"\u007f", r"\\u0080", r"\\\uFFFF"]),),
        (r'["\u0061", "\u007f", "\u00e9"]',),
        (json.dumps(["中文", "😀", "é"], ensure_ascii=False),),
        (json.dumps(["\x00", "\x01", "\x1f"]),),
        ("broken",),
        ('["a",]',),
        ("{}",),
        ("null",),
        ("[1, null]",),
    ]
    assert check_function(packed, schema, backend, rows) == []
    program = trace(packed, schema)
    lowered = backend.lower_program(program)
    key = OpKey("json.dumps", (Kind.LIST, Kind.BOOL))
    rewrite = backend.rules.rewrite_call

    def previous_serializer(node: Call) -> Node | None:
        return serialized_strings(*node.arguments) if node.op.key == key else rewrite(node)

    with patch.object(backend.rules, "rewrite_call", previous_serializer):
        previous = backend.lower_program(program)
    assert lowered.exact is not None and previous.exact is not None
    current, original = backend.evaluate([lowered.exact, previous.exact], schema, rows)
    assert [flag is True for flag in current] == [flag is True for flag in original]
    assert current[1:5] == [True] * 4
    transform = "list_transform" if backend.name == "duckdb" else "array_transform"
    assert str(lowered.exact).count(transform) < str(previous.exact).count(transform)


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"duckdb", "datafusion"}]
)
@pytest.mark.parametrize("filtered", [False, True])
def test_json_changed_characters_keep_element_guards(backend: Backend[Any], filtered: bool) -> None:
    def packed(text: str) -> str:
        values = [" ".join(value.split(" ")) for value in json.loads(text)]
        if filtered:
            values = [value for value in values if value != "é"]
        else:
            values = ["é".join(sorted(value.split(" "))) for value in values]
        return json.dumps(values)

    key = OpKey("json.dumps", (Kind.LIST, Kind.BOOL))
    original = backend.rules.rewrite_call

    def serialize(node: Call) -> Node | None:
        result = original(node)
        if node.op.key == key:
            assert result == serialized_strings(*node.arguments)
        return result

    schema = {"text": STR}
    rows = [("[]",), ('["é", "a b"]',)]
    with patch.object(backend.rules, "rewrite_call", serialize):
        assert check_function(packed, schema, backend, rows) == []
