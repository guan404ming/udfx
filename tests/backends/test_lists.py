from typing import Any

import pytest
from conftest import backend_params
from pyspark.sql import functions

from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.backends.spark.dtypes import dtype
from udfx.frontend import trace
from udfx.ir import TRUE, Col, Comprehension
from udfx.testing import check_function
from udfx.types import STR, list_of


@pytest.mark.parametrize("backend", backend_params())
def test_comprehension_native_rows(backend: Backend[Any]) -> None:
    def mapped(words: list[str]) -> list[str]:
        return [word.upper() for word in words if word.startswith("a")]

    schema = {"words": list_of(STR)}
    program = trace(mapped, schema)
    assert isinstance(program.value, Comprehension)
    lowered = backend.lower(program.value)
    exact = lowered.exact if lowered.exact is not None else backend.literal(TRUE)
    results, flags = backend.evaluate([lowered.expr, exact], schema, [(["abc", "b"],), ([],)])
    if backend.name == "datafusion":
        assert flags == [False, False]
    else:
        assert results == [["ABC"], []]
        assert flags == [True, True]


@pytest.mark.parametrize("backend", backend_params())
def test_comprehension_membership_fallback(backend: Backend[Any]) -> None:
    def filtered(words: list[str | None]) -> list[str | None]:
        return [word for word in words if word not in ("a", None)]

    schema = {"words": list_of(STR.or_none()).or_none()}
    rows = [(["a", "b", None],), ([],), (None,)]
    assert check_function(filtered, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [any, all], ids=["any", "all"])
def test_nullable_list_reductions(backend: Backend[Any], function: Any) -> None:
    def reduced(words: list[str] | None) -> bool:
        return function(words)

    schema = {"words": list_of(STR).or_none()}
    assert check_function(reduced, schema, backend, [(None,), ([],), (["a", ""],)]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_set_iteration_uses_python_order(backend: Backend[Any]) -> None:
    def copied(words: list[str]) -> list[str]:
        return list(set(words))

    schema = {"words": list_of(STR)}
    rows = [(["b", "a", "b"],), (["z", "Z", "a", ""],), ([],)]
    if isinstance(backend, SparkBackend):
        translation = compile(copied, schema, backend)
        assert translation.expr is not None
        reference = functions.udf(copied, dtype(list_of(STR)))(
            backend.column(Col("words", list_of(STR)))
        )
        actual, expected = backend.evaluate([translation.expr, reference], schema, rows)
        assert actual == expected
        return
    assert check_function(copied, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_constant_generator_unrolls_natively(backend: Backend[Any]) -> None:
    def matched(text: str) -> bool:
        return any(word in text for word in ("a", "b"))

    schema = {"text": STR}
    program = trace(matched, schema)
    lowered = backend.lower(program.value)
    exact = lowered.exact if lowered.exact is not None else backend.literal(TRUE)
    values, flags = backend.evaluate([lowered.expr, exact], schema, [("",), ("abc",), ("z",)])
    assert values == [False, True, False]
    assert flags == [True, True, True]
