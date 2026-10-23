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
from udfx.types import FLOAT, INT, STR, list_of


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
        reference = functions.udf(copied, dtype(list_of(STR)), useArrow=False)(
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


@pytest.mark.parametrize("backend", backend_params())
def test_numeric_sets_match_python(backend: Backend[Any]) -> None:
    def count(values: list[float]) -> int:
        return len(set(values))

    def ordered(values: list[float]) -> list[float]:
        return sorted(set(values))

    schema = {"values": list_of(FLOAT)}
    rows = [
        ([],),
        ([1.0, 2.0, 1.0],),
        ([-0.0, 0.0],),
        ([0.0, -0.0],),
        ([float("nan"), float("nan")],),
    ]
    assert check_function(count, schema, backend, rows) == []
    assert check_function(ordered, schema, backend, rows) == []
    lowered = backend.lower(trace(count, schema).value)
    assert lowered.exact is not None
    flags = backend.evaluate([lowered.exact], schema, rows)[0]
    assert flags == [True, True, False, False, False]


@pytest.mark.parametrize("backend", backend_params())
def test_set_algebra_native_rows(backend: Backend[Any]) -> None:
    def score(left: list[str], right: list[str]) -> float:
        first = set(left)
        second = set(right)
        return len(first & second) / len(first | second)

    schema = {"left": list_of(STR).or_none(), "right": list_of(STR).or_none()}
    rows = [
        (["a", "a", "b"], ["b", "c", "c"]),
        ([], ["a", "a"]),
        (["", "\x00", "😀"], ["😀", ""]),
    ]
    assert check_function(score, schema, backend, [*rows, ([], []), (None, []), ([], None)]) == []
    assert backend.lower(trace(score, schema).value).exact is None


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("enumerated", [False, True], ids=["direct", "enumerated"])
def test_dynamic_range_native_rows(backend: Backend[Any], enumerated: bool) -> None:
    def increments(stop: int) -> list[int]:
        if enumerated:
            return [index + 1 for index, _value in enumerate(range(stop))]
        return [index + 1 for index in range(stop)]

    schema = {"stop": INT}
    rows = [(-2,), (0,), (3,)]
    assert check_function(increments, schema, backend, rows) == []
    lowered = backend.lower(trace(increments, schema).value)
    assert lowered.exact is not None
    flags = backend.evaluate([lowered.exact], schema, rows)[0]
    assert flags == [True] * 3


@pytest.mark.parametrize("backend", backend_params())
def test_constant_paired_comprehension(backend: Backend[Any]) -> None:
    def selected(text: str) -> list[str]:
        return [value for _index, value in enumerate([text, "fixed"])]

    schema = {"text": STR}
    assert check_function(selected, schema, backend, [("",), ("a",)]) == []
    if backend.name in {"duckdb", "spark"}:
        lowered = backend.lower(trace(selected, schema).value)
        exact = lowered.exact
        assert exact is None or backend.evaluate([exact], schema, [("a",)])[0] == [True]


def indexed_words(text: str) -> list[str]:
    return [str(index) + word for index, word in enumerate(text.split()) if word != "a"]


def weighted_lengths(text: str) -> list[int]:
    return [len(word) * weight for word, weight in zip(text.split(), (1, 2), strict=False)]


def weighted_numbers(text: str) -> list[int]:
    return [int(word) * weight for word, weight in zip(text.split(), (1, 2), strict=False)]


def keyed_values(text: str) -> list[str]:
    return [key + str(value) for key, value in {"a": 1, "b": 2}.items() if key in text]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function", [indexed_words, weighted_lengths, weighted_numbers, keyed_values]
)
def test_indexed_comprehension_native_rows(backend: Backend[Any], function: Any) -> None:
    schema = {"text": STR}
    rows = [("",), ("7",), ("12 3",), ("1 2 a",)]
    assert check_function(function, schema, backend, [*rows, ("x",)]) == []
    lowered = backend.lower(trace(function, schema).value)
    exact = lowered.exact if lowered.exact is not None else backend.literal(TRUE)
    assert backend.evaluate([lowered.expr, exact], schema, rows)[1] == [True] * len(rows)
