import json

import pyarrow as pa
import pytest
from datafusion import SessionContext, col, functions, lit

from udfx.backends.datafusion import DataFusionBackend
from udfx.backends.datafusion.rules import ARRAY_SCOPING, array_transform
from udfx.frontend import trace
from udfx.semantics.text import whitespace_normalization_pattern
from udfx.types import STR, list_of


def test_nested_array_transform_uses_one_scope_wrapper() -> None:
    source = pa.table({"value": [[["a", "b"], []]]})
    expression = array_transform(
        col("value"),
        lambda items: array_transform(
            items.cast(pa.list_(pa.string())), lambda item: item.cast(pa.string())
        ),
    )
    assert str(expression).count("array_transform") == 3
    output = SessionContext().from_arrow(source).select(expression).to_arrow_table()
    assert output.column(0).to_pylist() == [[["a", "b"], []]]


def test_json_encoding_uses_three_replacement_passes() -> None:
    program = trace(lambda value: json.dumps(value, ensure_ascii=False), {"value": STR})
    lowered = DataFusionBackend().lower_program(program)
    assert lowered.exact is None
    assert str(lowered.expr).count("replace(") == 3


def test_word_split_skips_identity_replacements() -> None:
    backend = DataFusionBackend()
    schema = {"text": STR}
    pattern = whitespace_normalization_pattern()
    expression = backend.build(trace(lambda text: text.split(), schema))
    assert pattern in str(expression)
    matches = functions.regexp_count(col("text"), lit(pattern))
    rows = [("one space",), ("two  spaces",), ("a \t\u3000 b",)]
    assert backend.evaluate([matches], schema, rows) == [[0, 1, 1]]


@pytest.mark.parametrize("nullable", [False, True])
def test_json_decoding_uses_native_parser(nullable: bool) -> None:
    def decoded(value: str) -> object:
        return json.loads(value)

    result = STR.or_none() if nullable else STR
    program = trace(decoded, {"value": STR}, result=result)
    expression = str(DataFusionBackend().lower_program(program).expr)
    assert "json_tuple(" in expression
    assert "replace(" not in expression


def test_supported_lambda_plans_do_not_add_array_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = DataFusionBackend()
    monkeypatch.setattr(backend, "supports_graphs", lambda: True)
    schema = {"values": list_of(STR)}
    program = trace(lambda values: [len(value) for value in values], schema)
    expression = backend.build(program)
    assert str(expression).count("array_transform") == 1
    assert backend.evaluate([expression], schema, [(["a", "bc"],), ([],)]) == [[[1, 2], []]]
    assert ARRAY_SCOPING.get() is True
