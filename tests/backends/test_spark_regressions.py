from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import pytest
from conftest import backend_params
from corpus.cases_lists import indexed_words, unpack_ends
from pyspark.sql import functions

from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.backends.spark.splitting import split_words
from udfx.ir import Col
from udfx.program import Program
from udfx.types import BOOL, INT, STR


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
    assert str(expr).count("regexp_extract_all") == 2
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
