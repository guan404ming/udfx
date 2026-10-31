import json
from unittest.mock import patch

import pytest
from conftest import spark_session
from pyspark.sql import Column, functions

from udfx.backends.diagnostics import capture_lowering
from udfx.backends.lowering_context import lowering_context, runtime_udfs
from udfx.backends.spark import SparkBackend, runtime
from udfx.backends.spark import json as spark_json
from udfx.frontend import trace
from udfx.ir import Col, lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import BOOL, STR, list_of

pytestmark = pytest.mark.xdist_group("spark")


@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_json_strings_runtime_preserves_nullable_lists(ensure_ascii: bool) -> None:
    backend = SparkBackend(spark_session(ansi=True).newSession())
    values = Col("values", list_of(STR.or_none()).or_none())
    rows = [(None,), ([],), ([None, "A", 'quote" slash\\', "\x00\x7f é 😀"],)]
    with (
        patch.object(
            spark_json, "encode_strings", side_effect=AssertionError("Unexpected string encoder")
        ),
        lowering_context(),
    ):
        lowered = backend.lower(apply("json.dumps", values, lit(ensure_ascii)))
        assert runtime_udfs() == ("spark.json_strings",)
    assert lowered.exact is None
    assert backend.evaluate([lowered.expr], {"values": values.type}, rows) == [
        [json.dumps(value, ensure_ascii=ensure_ascii) for (value,) in rows]
    ]


def test_json_strings_runtime_accepts_dynamic_ascii_and_null_faults() -> None:
    backend = SparkBackend(spark_session(ansi=True).newSession())
    values = [None, "é 😀", "\b\t\n\f\r", "\x01\x7f"]
    rows = [(value, ascii) for value in (None, [], values) for ascii in (False, True, None)]
    with runtime.scope(backend.session):
        expr = runtime.call("json_strings", functions.col("values"), functions.col("ascii"))
    assert expr is not None
    schema = {"values": list_of(STR.or_none()).or_none(), "ascii": BOOL.or_none()}
    assert backend.evaluate([expr], schema, rows) == [
        [None if ascii is None else json.dumps(value, ensure_ascii=ascii) for value, ascii in rows]
    ]


def test_json_ascii_uppercase_free_values_keep_engine_encoding() -> None:
    def packed(value: str) -> str:
        return json.dumps([value.lower()])

    backend = SparkBackend(spark_session(ansi=True).newSession())
    rows = [("ABC",), ("ÉCOLE 漢 😀",), (r"\uABCD",)]
    with (
        patch.object(spark_json, "call", side_effect=AssertionError("Unexpected runtime kernel")),
        capture_lowering() as details,
    ):
        assert check_function(packed, {"value": STR}, backend, rows) == []
    assert details[-1].runtime_udfs == ()


def test_json_unicode_uppercase_free_values_use_runtime_encoding() -> None:
    def packed(value: str) -> str:
        return json.dumps([value.lower()], ensure_ascii=False)

    backend = SparkBackend(spark_session(ansi=True).newSession())
    rows = [("ABC",), ("ÉCOLE 漢 😀",), (r"\uABCD",)]
    with (
        patch.object(
            spark_json, "encode_strings", side_effect=AssertionError("Unexpected string encoder")
        ),
        capture_lowering() as details,
    ):
        assert check_function(packed, {"value": STR}, backend, rows) == []
    assert details[-1].runtime_udfs == ("spark.json_strings",)


def test_json_word_source_keeps_its_fused_runtime_kernel() -> None:
    def packed(value: str) -> str:
        return json.dumps(value.split())

    backend = SparkBackend(spark_session(ansi=True).newSession())
    original = runtime.call
    names: list[str] = []

    def call(name: str, *arguments: Column) -> Column | None:
        assert name == "json_words"
        names.append(name)
        return original(name, *arguments)

    with patch.object(spark_json, "call", call), capture_lowering() as details:
        expr = backend.build(trace(packed, {"value": STR}))
    assert names
    assert details[-1].runtime_udfs == ("spark.json_words",)
    assert backend.evaluate([expr], {"value": STR}, [(" a\t😀 ",)]) == [[json.dumps(["a", "😀"])]]
