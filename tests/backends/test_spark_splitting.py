from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import INT, INT64_MAX, INT64_MIN, STR


@pytest.mark.parametrize(
    "function",
    [lambda text, separator: text.split(separator), lambda text, separator: text.rsplit(separator)],
)
def test_absent_separator_skips_splitting(function: Callable[..., object]) -> None:
    backend = SparkBackend()
    schema = {"text": STR, "separator": STR.or_none()}
    expression = str(backend.lower(trace(function, schema).value).expr)
    assert "contains(text, separator)" in expression


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("separator", [None, " ", ".", "*", "\\E", "[]", "aa", "😀", "\x00"])
def test_literal_separator_skips_presence_scan(
    backend: Backend[Any], separator: str | None
) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark literal split")

    def split(text: str) -> list[str]:
        return text.split(separator)

    schema = {"text": STR.or_none()}
    expression = str(backend.lower(trace(split, schema).value).expr)
    assert "contains(" not in expression
    assert "regexp_replace(" not in expression
    marker = separator or " "
    rows = [
        (text,) for text in (None, "", "plain", marker, marker.join(("", "\u03b1", "", "😀", "")))
    ]
    assert check_function(split, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda text, maximum: text.split("|", maximum),
        lambda text, maximum: text.rsplit("|", maximum),
    ],
)
def test_spark_split_limit_skips_array_reconstruction(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark split limit")
    schema = {"text": STR, "maximum": INT}
    expression = str(backend.lower(trace(function, schema).value).expr)
    assert "array_append(" not in expression
    assert "array_join(" not in expression
    rows = [
        (text, maximum)
        for text in ("", "a||b|", "😀|ß|ƛ")
        for maximum in (INT64_MIN, -1, 0, 1, 2**31 - 2, 2**31 - 1, INT64_MAX)
    ]
    assert check_function(function, schema, backend, rows) == []


def test_spark_last_part_skips_array_materialization() -> None:
    def tail(text: str) -> str:
        return text[text.rfind("-") + 1 :]

    backend = SparkBackend()
    schema = {"text": STR}
    expression = str(backend.lower(trace(tail, schema).value).expr)
    assert "substring_index" in expression
    assert "split_part" not in expression
    rows = [(text,) for text in ("", "abc", "-a--b-", "😀-ß-ä", "a\x00-b")]
    assert check_function(tail, schema, backend, rows) == []


@pytest.mark.parametrize("separator", ["|", "😀", "]", "||"])
@pytest.mark.parametrize("step", [-3, -2, 2, 3, 65])
def test_split_stride_preserves_empty_tokens(separator: str, step: int) -> None:
    def sliced(text: str, start: int | None, stop: int | None) -> list[str]:
        return text.split(separator)[start:stop:step]

    backend = SparkBackend()
    schema = {"text": STR.or_none(), "start": INT.or_none(), "stop": INT.or_none()}
    expression = str(backend.lower(trace(sliced, schema).value).expr)
    assert ("filter(" in expression) == (len(separator) != 1 or abs(step) > 64)
    rows = [
        (text, start, stop)
        for text in ("", "a", separator, separator.join(("", "a", "", "b", "")))
        for start in (None, INT64_MIN, -4, -1, 0, 2, INT64_MAX)
        for stop in (None, INT64_MIN, -2, 0, 3, INT64_MAX)
    ]
    assert check_function(sliced, schema, backend, rows) == []
    assert check_function(sliced, schema, backend, [(None, None, None)]) == []


@pytest.mark.parametrize("step", [-2, 2])
def test_static_split_stride_keeps_array_filter(step: int) -> None:
    def sliced(text: str) -> list[str]:
        return text.split("|")[::step]

    backend = SparkBackend()
    schema = {"text": STR}
    expression = str(backend.lower(trace(sliced, schema).value).expr)
    assert "filter(" in expression
    rows = [(text,) for text in ("", "a", "|", "|a||b|")]
    assert check_function(sliced, schema, backend, rows) == []


def test_right_split_keeps_overlapping_separators() -> None:
    def split(text: str, separator: str | None, maximum: int) -> list[str]:
        return text.rsplit(separator, maximum)

    backend = SparkBackend()
    schema = {"text": STR, "separator": STR.or_none(), "maximum": INT}
    expression = str(backend.lower(trace(split, schema).value).expr)
    assert "substring_index(" in expression
    rows = [
        (text, separator, maximum)
        for text in ("", "aaaaa", "abXabXab", "a😀b😀", "x::y::z", " a\tb c ")
        for separator in (None, "", "aa", "abXab", "😀", "::", "b", "x:")
        for maximum in (INT64_MIN, -1, 0, 1, 2, INT64_MAX)
    ]
    assert check_function(split, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_unlimited_word_split_skips_reversal(backend: Backend[Any]) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark word split")

    def words(text: str, maximum: int) -> list[str]:
        return text.rsplit(None, maximum)

    schema = {"text": STR.or_none(), "maximum": INT}
    rows = [
        (text, maximum)
        for text in (None, "", "  ", " a  b c\t", "\u001cß\u2000😀\u2029", "a b " * 5000)
        for maximum in (INT64_MIN, -1, 0, 1, 3, 2000, INT64_MAX)
    ]
    assert check_function(words, schema, backend, rows) == []
    expression = backend.build(trace(lambda text: text.rsplit(None, -1), {"text": STR}))
    frame = backend.session().createDataFrame([("a b",)], ["text"])
    output = StringIO()
    with redirect_stdout(output):
        frame.select(expression.alias("result")).explain()
    assert "reverse(" not in output.getvalue()
