from collections.abc import Callable
from contextlib import redirect_stdout
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
from udfx.ir import Col, Guarded, lit
from udfx.ops import apply
from udfx.program import Program
from udfx.types import BOOL, INT, STR, list_of


def test_spark_uses_public_apis() -> None:
    for path in Path(spark.__file__).parent.rglob("*.py"):
        source = path.read_text()
        for name in ("_jvm", "_jc", "_jsparkSession", "py4j", "catalyst"):
            assert name not in source, f"{path.name}: {name}"


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
