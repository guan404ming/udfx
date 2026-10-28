import json
from collections.abc import Mapping
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import pytest
from conftest import backend_params
from pyspark.sql import Column, functions

from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.backends.spark.json import typed_field
from udfx.backends.spark.splitting import split_words
from udfx.frontend import trace
from udfx.ir import Col, Node
from udfx.program import Program
from udfx.testing import check_function
from udfx.types import BOOL, INT, STR, list_of
from workloads.corpus.cases_lists import indexed_words, unpack_ends


@pytest.mark.parametrize("backend", backend_params())
def test_spark_fallback_returns_only_python_result(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark plan")
    program = Program(
        lambda text, exact: len(text.split()),
        {"text": STR, "exact": BOOL},
        Col("value", INT),
    )
    native = functions.size(split_words(backend.column(Col("text", STR))))
    expr = backend.fallback(native, functions.col("exact"), program)
    assert str(expr).count("regexp_extract_all") == str(native).count("regexp_extract_all")
    frame = backend.session().createDataFrame([("one two", True)], ["text", "exact"])
    output = StringIO()
    with redirect_stdout(output):
        frame.select(expr.alias("result")).explain()
    plan = output.getvalue()
    assert plan.count("ArrowEvalPython") == 1
    projection = next(line for line in plan.splitlines() if "Project [" in line)
    assert "pythonUDF" in projection
    assert "CASE WHEN" not in projection
    assert "regexp_extract_all" not in projection


@pytest.mark.parametrize("backend", backend_params())
def test_spark_unpack_ends_evaluates_one_whitespace_split(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark plan")
    translation = compile(unpack_ends, {"text": STR.or_none()}, backend)
    assert translation.ok, translation.unsupported
    assert translation.expr is not None
    frame = backend.session().createDataFrame([("one two three",)], ["text"])
    output = StringIO()
    with redirect_stdout(output):
        frame.select(translation.expr.alias("result")).explain()
    plan = output.getvalue()
    assert "regexp_extract(" not in plan
    assert "regexp_replace" not in plan
    assert "Python" not in plan


@pytest.mark.parametrize("backend", backend_params())
def test_spark_indexed_comprehension_splits_once_per_use(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark plan")
    translation = compile(indexed_words, {"text": STR.or_none()}, backend)
    assert translation.ok, translation.unsupported
    assert str(translation.expr).count("regexp_extract_all") <= 8


@pytest.mark.parametrize("backend", backend_params())
def test_spark_membership_uses_search(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark plan")

    def contains(items: list[str | None], needle: str | None) -> bool:
        return needle in items

    schema = {"items": list_of(STR.or_none()), "needle": STR.or_none()}
    expression = str(backend.lower(trace(contains, schema).value).expr)
    assert "array_intersect" not in expression
    assert "array_contains" in expression
    rows = [(["a", None], "x"), ([None], None), ([], None), (["x"], "x")]
    assert check_function(contains, schema, backend, rows) == []


def guarded_counts(text: str) -> list[str]:
    words = text.split(",")
    return [word.capitalize() + str(len(words)) for word in words]


def faulted_counts(text: str) -> list[str]:
    words = text.split(",")
    return [word[1] + str(len(words)) for word in words]


def guarded_outer_counts(text: str) -> list[int]:
    words = text.split(",")
    return [len(words) for _ in words] if text.capitalize() == text else [0]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [guarded_counts, faulted_counts, guarded_outer_counts])
def test_spark_guarded_comprehension_keeps_fallback(backend: Backend[Any], function: Any) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark plan")
    schema = {"text": STR}
    lowered = backend.lower(trace(function, schema).value)
    assert backend.exact(lowered) is not None
    rows = [("AB,CD",), ("ab,cd",), ("ɤ,ß",), ("",)]
    assert check_function(function, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_spark_fallback_skips_inexact_native_values(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark plan")
    schema = {"value": INT, "exact": BOOL.or_none()}
    program = Program(lambda value, exact: value, schema, Col("value", INT))
    native = functions.when(functions.col("exact"), functions.col("value")).otherwise(
        functions.raise_error("inexact native value")
    )
    expression = backend.fallback(native, functions.col("exact"), program)
    assert backend.evaluate([expression], schema, [(1, True), (2, False), (3, None)]) == [[1, 2, 3]]


@pytest.mark.parametrize("backend", backend_params())
def test_spark_json_field_binding_keeps_outer_keys(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark lambda capture")

    def selected(documents: list[str], exact: bool) -> list[str | None]:
        return [json.loads(document).get(document[0]) for document in documents]

    schema = {"documents": list_of(STR), "exact": BOOL}
    program = Program(selected, schema, Col("result", list_of(STR.or_none())))
    source = backend.column(Col("documents", schema["documents"]))

    def fields(document: Column) -> tuple[Column, Column, Column]:
        return typed_field(document, functions.substring(document, 1, 1), STR.or_none(), ())

    native = functions.transform(source, lambda document: fields(document)[0])
    exact = functions.forall(source, lambda document: fields(document)[2]) & functions.col("exact")
    expression = backend.fallback(native, exact, program)
    rows = [([], True), (['{"{":"native"}', "{}"], True), (['{"{":"python"}'], False)]
    assert backend.evaluate([expression], schema, rows) == [program.run(rows)]


@pytest.mark.parametrize("backend", backend_params())
def test_spark_lowers_unbound_fallback_once(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark lowering")
    program = trace(lambda text: text.title(), {"text": STR})
    lower = backend.lower_node
    calls = 0

    def record(node: Node, elements: Mapping[Node, Any]) -> Any:
        nonlocal calls
        calls += int(node is program.value and not elements)
        return lower(node, elements)

    monkeypatch.setattr(backend, "lower_node", record)
    backend.build(program)
    assert calls == 1
