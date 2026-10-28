import json
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from conftest import backend_params
from pyspark.sql import functions

from udfx import compile
from udfx.backends import spark
from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.backends.spark.json import ENCODING, typed_field
from udfx.ir import Col, Guarded, lit
from udfx.ops import apply
from udfx.program import Program
from udfx.semantics.json_encoding import encode_strings
from udfx.types import BOOL, DATETIME, INT, STR, list_of


def test_spark_uses_public_apis() -> None:
    for path in Path(spark.__file__).parent.rglob("*.py"):
        source = path.read_text()
        for name in ("_jvm", "_jc", "_jsparkSession", "py4j", "catalyst"):
            assert name not in source, f"{path.name}: {name}"


def test_character_split_skips_array_scan() -> None:
    def characters(value: str) -> list[str]:
        return list(value)

    translation = compile(characters, {"value": STR}, SparkBackend())
    assert translation.ok
    assert translation.expr is not None
    assert "array_remove" not in str(translation.expr)


def test_datetime_difference_skips_epoch_conversion() -> None:
    backend = SparkBackend()
    difference = apply("datetime_difference", Col("left", DATETIME), Col("right", DATETIME))
    expr = str(backend.lower(difference).expr).lower()
    assert "1970" not in expr
    assert "timestampdiff" not in expr
    assert "timestamp_diff" not in expr


def test_json_validation_skips_variant_serialization() -> None:
    backend = SparkBackend()
    lowered = backend.lower(apply("json.valid", Col("value", STR)))
    expr = str(backend.exact_expression(lowered))
    assert "get_json_object" in expr
    assert "try_parse_json" in expr
    assert "to_json" not in expr


@pytest.mark.parametrize("nullable", [False, True])
def test_json_projection_binds_selected_variant(nullable: bool) -> None:
    backend = SparkBackend()
    value = backend.column(Col("value", STR))
    key = backend.column(Col("key", STR))
    result = STR.or_none() if nullable else STR
    expressions = typed_field(value, key, result, (0,))
    assert all(str(expression).count("try_parse_json") == 1 for expression in expressions)


def test_json_encoding_skips_unused_escape_normalization() -> None:
    backend = SparkBackend()
    schema = {"value": list_of(STR.or_none()).or_none()}
    primitives = replace(
        ENCODING,
        tokens=lambda value, pattern: functions.raise_error("unused normalization").cast(
            "array<string>"
        ),
    )
    expr = encode_strings(primitives, backend.column(Col("value", schema["value"])), True)
    values = [None, [], [None, "null", ""], ['a"b', "a\\b", "\n\t", "normal"]]
    assert backend.evaluate([expr], schema, [(value,) for value in values]) == [
        [json.dumps(value) for value in values]
    ]


@pytest.mark.parametrize("backend", backend_params())
def test_constant_exact_guard_removes_python(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark plan")
    session = backend.session()
    program = Program(
        lambda value: value,
        {"value": INT},
        Guarded(Col("value", INT), apply("__eq__", lit(1), lit(1))),
    )
    expr = backend.build(program)
    frame = session.range(3).selectExpr("id AS value").select(expr)
    output = StringIO()
    with redirect_stdout(output):
        frame.explain()
    assert "Python" not in output.getvalue()


@pytest.mark.parametrize("backend", backend_params())
def test_fallback_preserves_aggregate_codegen(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark plan")
    frame = backend.session().createDataFrame([([1, 2], True), ([3, 4], False)], ["items", "exact"])
    program = Program(
        lambda items, exact: len(items),
        {"items": list_of(INT), "exact": BOOL},
        Col("value", INT),
    )
    native = functions.size("items").cast("long")
    expr = backend.fallback(native, functions.col("exact"), program)
    selected = frame.select(functions.sum(expr))
    selected.collect()
    output = StringIO()
    with redirect_stdout(output):
        selected.explain("codegen")
    assert "HashAggregate" in output.getvalue()


@pytest.mark.parametrize("backend", backend_params())
def test_fallback_uses_one_guard(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark plan")
    schema = {"value": STR, "other": STR}
    frame = backend.session().createDataFrame([("value", "other")], list(schema))
    program = Program(lambda value, other: value, schema, Col("value", STR))
    exact = functions.octet_length("value") == functions.length("value")
    expression = backend.fallback(functions.col("value"), exact, program)
    output = StringIO()
    with redirect_stdout(output):
        frame.select(expression.alias("result")).explain()
    assert output.getvalue().count("octet_length(") == 1


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: value[-4:4],
        lambda value: value[-4:4:2],
        lambda value: value[4:-4:-1],
        lambda value: value[4:-4:-2],
        lambda value: value[-64:64],
    ],
    ids=["slice", "stride", "reverse-slice", "reverse-stride", "bound"],
)
def test_bound_strings_use_codegen(
    backend: Backend[Any], function: Callable[[str], object]
) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark plan")
    translation = compile(function, {"value": STR}, backend)
    assert translation.ok
    assert translation.expr is not None
    assert str(translation.expr).count("value") == 1
    frame = backend.session().createDataFrame([("text",)], ["value"])
    selected = frame.select(translation.expr)
    output = StringIO()
    with redirect_stdout(output):
        selected.explain("codegen")
    assert "Project [" in output.getvalue()


@pytest.mark.parametrize("backend", backend_params())
def test_literal_split_uses_codegen(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark plan")
    translation = compile(
        lambda value, separator: value.split(separator), {"value": STR, "separator": STR}, backend
    )
    assert translation.ok
    assert translation.expr is not None
    frame = backend.session().createDataFrame([("a,b", ",")], ["value", "separator"])
    selected = frame.select(translation.expr)
    output = StringIO()
    with redirect_stdout(output):
        selected.explain("codegen")
    assert "Project [" in output.getvalue()


@pytest.mark.parametrize("separator", [",", ".", "[", "\\", "😀"])
def test_single_character_split_keeps_compact_patterns(separator: str) -> None:
    backend = SparkBackend()
    translation = compile(lambda text: text.split(separator), {"text": STR}, backend)
    assert translation.ok
    assert translation.expr is not None
    frame = backend.session().createDataFrame([("a,b",)], ["text"])
    output = StringIO()
    with redirect_stdout(output):
        frame.select(translation.expr.alias("result")).explain("extended")
    optimized = output.getvalue().split("== Optimized Logical Plan ==", 1)[1]
    optimized = optimized.split("== Physical Plan ==", 1)[0]
    assert r"\Q" not in optimized
    assert "regexp_replace" not in optimized


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(("start", "stop"), [(-65, 64), (-64, 65), (64, -65)])
def test_large_mixed_slices_use_binding(backend: Backend[Any], start: int, stop: int) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark plan")
    step = -1 if start > 0 else 1
    translation = compile(lambda value: value[start:stop:step], {"value": STR}, backend)
    assert translation.ok
    expr = str(translation.expr)
    assert expr.count("value") == 1
    assert "regexp_extract" not in expr
    assert "transform(" in expr
