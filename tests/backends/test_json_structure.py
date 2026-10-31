import sys
from typing import Any

import pytest
from conftest import backend_params
from pyspark.sql import functions

from udfx.backends.base import Backend
from udfx.backends.spark import json as spark_json
from udfx.backends.spark.conversions import TEXT
from udfx.backends.spark.json import replace
from udfx.ir import Col
from udfx.semantics import json_validation
from udfx.semantics.json_lowering import accepted
from udfx.semantics.json_structure import complete_integer_document
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
def test_json_container_completeness_skips_serialization(backend: Backend[Any]) -> None:
    if backend.name != "spark":
        pytest.skip("Spark JSON completeness")
    schema = {"text": STR}
    value = backend.column(Col("text", STR))
    serialized = functions.raise_error("Unexpected JSON serialization").cast("string")
    complete = complete_integer_document(TEXT, replace, value, serialized)
    rows = [("[0,-0,1,-1]",), ('{"value":["x",null,true]}',), ("[" * 32 + "0" + "]" * 32,)]
    assert backend.evaluate([complete], schema, rows) == [[True] * len(rows)]


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"spark", "spark-ansi-off"}]
)
def test_json_container_proof_skips_redundant_checks(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        json_validation,
        "within_limits",
        lambda *arguments: functions.raise_error("Unexpected JSON limit scan").cast("boolean"),
    )
    monkeypatch.setattr(
        spark_json,
        "complete_integer_document",
        lambda *arguments: functions.raise_error("Unexpected JSON structure scan").cast("boolean"),
    )
    schema = {"text": STR}
    value = backend.column(Col("text", STR))
    rows = [
        ('{"value":["x",null,true,123]}',),
        ('{"value":"\\ud83d\\ude00"}',),
        ("[" * 32 + "0" + "]" * 32,),
        ("[" + ",".join(["[0]"] * 500) + "]",),
    ]
    assert backend.evaluate([accepted(spark_json.NATIVE, value)], schema, rows) == [
        [True] * len(rows)
    ]


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"spark", "spark-ansi-off"}]
)
def test_json_container_proof_respects_limits(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "get_int_max_str_digits", lambda: 640)
    schema = {"text": STR}
    value = backend.column(Col("text", STR))
    proven = value.rlike(json_validation.bounded_integer_container())
    rows = [
        ("[" + "1" * 640 + "]",),
        ("[" + "1" * 641 + "]",),
        ("[" * 32 + "0" + "]" * 32,),
        ("[" * 33 + "0" + "]" * 33,),
    ]
    assert backend.evaluate([proven], schema, rows) == [[True, False, True, False]]
