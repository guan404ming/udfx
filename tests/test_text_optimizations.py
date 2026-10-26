import re
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
from conftest import backend_params

from bench.inspection import fallback_in_plan
from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.duckdb import DuckDBBackend
from udfx.backends.polars import PolarsBackend
from udfx.backends.program_lowering import lower_program
from udfx.backends.spark import SparkBackend
from udfx.frontend import trace
from udfx.frontend.core.text import combined_search
from udfx.ir import Call, Col, lit
from udfx.ops import apply
from udfx.semantics.text import ANY_CHAR, char_class, whitespace
from udfx.semantics.text_predicates import ascii_free
from udfx.testing import check_function
from udfx.testing.text_traps import STRING_TRAPS
from udfx.types import STR
from workloads.corpus.cases_regex import first_number


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: value.upper().lower(),
        lambda value: value.lower().capitalize().casefold(),
    ],
)
def test_ascii_case_chain(backend: Backend[Any], function: Callable[[str], str]) -> None:
    schema = {"value": STR}
    program = trace(function, schema)
    lowered = backend.lower(program.value)
    assert lowered.exact is not None
    rows = [("AbC" * 10,), ("",), ("ß",), ("İ",), ("ΟΣ",)]
    assert backend.evaluate([lowered.exact], schema, rows)[0] == [True, True, False, False, False]
    assert check_function(function, schema, backend, rows) == []


@pytest.mark.parametrize(
    ("cluster", "exact"),
    [
        pytest.param("a\u0301", False, id="extend"),
        pytest.param("a\uff9e", False, id="extend-halfwidth"),
        pytest.param("a\u0903", False, id="spacing-mark"),
        pytest.param("\u0600a", False, id="prepend"),
        pytest.param("👩\u200d💻", False, id="zwj"),
        pytest.param("🇺🇸", False, id="regional-indicators"),
        pytest.param("\u1100\u1161\u11a8", False, id="hangul-jamo"),
        pytest.param("\uac00\u1161", False, id="hangul-lv"),
        pytest.param("\uac01\u11a8", False, id="hangul-lvt"),
        pytest.param("\U00016d63\U00016d67", False, id="kirat-rai-vowels"),
        pytest.param("a\u1acf", False, id="unicode-17-extend"),
        pytest.param("a\U00011b61", False, id="unicode-17-spacing-mark"),
        pytest.param("\r\n", False, id="crlf"),
        pytest.param("\uac00", True, id="single-hangul-syllable"),
        pytest.param("a", True, id="ascii"),
    ],
)
def test_polars_reverse_cluster_guard(cluster: str, exact: bool) -> None:
    def function(text: str) -> str:
        return text[::-1]

    text = cluster + "x"
    assert pl.select(pl.lit(text).str.reverse()).item() == "x" + cluster
    assert (text[::-1] == "x" + cluster) is exact
    backend = PolarsBackend()
    schema = {"text": STR}
    lowered = backend.lower(trace(function, schema).value)
    assert lowered.exact is not None
    assert backend.evaluate([lowered.exact], schema, [(text,)]) == [[exact]]
    assert check_function(function, schema, backend, [(text,)]) == []


def test_spark_case_guard_skips_regex_for_ascii() -> None:
    schema = {"value": STR.or_none()}
    backend = SparkBackend()
    lowered = lower_program(backend, trace(lambda value: value.upper(), schema))
    assert lowered.exact is None
    assert "octet_length(value)" in str(lowered.expr)
    assert "CASE WHEN" in str(lowered.expr)
    assert "patch(" not in str(lowered.expr)
    rows = [("abc",), ("ƛ",), (None,), ("okɤ",)]
    assert check_function(lambda value: value.upper(), schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function", [lambda text: text.upper(), lambda text: text.lower(), lambda text: text.title()]
)
def test_spark_case_mapping_stays_native(
    backend: Backend[Any], function: Callable[[str], str]
) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark case mapping")
    schema = {"text": STR.or_none()}
    translation = compile(function, schema, backend)
    assert translation.ok, translation.unsupported
    assert not fallback_in_plan(translation.expr, "spark")
    points = ("", "Abßﬃ", "ΣΟΣ İı \u017f", "ǆƛɤ𐐨", "\u1c89", "\U00010d70", "\u1ad0")
    values = [
        None,
        *points,
        *(f"AΣ{point}A" for point in points),
        *(f"A{point}Σ" for point in points),
    ]
    assert check_function(function, schema, backend, [(value,) for value in values]) == []


def test_ascii_free_classes() -> None:
    assert ascii_free(char_class("Σİ"))
    assert not ascii_free(char_class("aΣ"))
    assert not ascii_free(ANY_CHAR)


def test_spark_split_uses_ascii_pattern_for_ascii() -> None:
    def words(value: str) -> list[str]:
        return value.split()

    schema = {"value": STR.or_none()}
    backend = SparkBackend()
    lowered = lower_program(backend, trace(words, schema))
    assert "octet_length(value)" in str(lowered.expr)
    rows = [("a b\tc",), ("a\xa0b",), ("a\u3000b\x1cc",), ("x\x85y\u2007z",), ("",), (None,)]
    assert check_function(words, schema, backend, rows) == []


def test_spark_regex_uses_ascii_classes_for_ascii() -> None:
    def symbols(value: str) -> str:
        return re.sub(r"[^\w\s]", "", value)

    schema = {"value": STR.or_none()}
    backend = SparkBackend()
    lowered = lower_program(backend, trace(symbols, schema))
    assert "octet_length(value)" in str(lowered.expr)
    rows = [("a,b!",), ("\xe9,\xfc_\u0663",), ("\u3000x.",), ("",), (None,)]
    assert check_function(symbols, schema, backend, rows) == []


def test_spark_search_runs_one_regex() -> None:
    schema = {"text": STR.or_none()}
    backend = SparkBackend()
    lowered = lower_program(backend, trace(first_number, schema))
    assert "regexp_like" not in str(lowered.expr)
    rows = [("a 12 b",), ("none",), ("\u0663\u0664 x",), ("",), (None,)]
    assert check_function(first_number, schema, backend, rows) == []


def test_duckdb_case_guard_skips_regex_for_ascii() -> None:
    def lower(value: str) -> str:
        return value.lower()

    backend = DuckDBBackend()
    schema = {"value": STR}
    lowered = backend.lower(trace(lower, schema).value)
    assert lowered.exact is not None
    assert "strlen" in str(lowered.exact)
    rows = [("AbC",), ("",), ("\x00a\n",)]
    assert backend.evaluate([lowered.exact], schema, rows) == [[True, True, True]]
    assert check_function(lower, schema, backend, rows + [(value,) for value in STRING_TRAPS]) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function, method",
    [
        (lambda value: value.strip().title(), "strip"),
        (lambda value: value.lstrip().title(), "lstrip"),
        (lambda value: value.rstrip().title(), "rstrip"),
    ],
)
def test_title_moves_before_whitespace_trim(
    backend: Backend[Any], function: Callable[[str], str], method: str
) -> None:
    schema = {"value": STR}
    program = trace(function, schema)
    assert isinstance(program.value, Call)
    assert program.value.op.name == f".{method}"
    values = [f"{whitespace()}{value}{whitespace()}" for value in STRING_TRAPS]
    assert check_function(function, schema, backend, [(value,) for value in values]) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("needle", ["noise", "k", "s", "i", "xi", "ix", "[a]", "", "ABC"])
def test_lowercase_search_preserves_unicode(backend: Backend[Any], needle: str) -> None:
    def contains(text: str) -> bool:
        return needle in text.lower()

    schema = {"text": STR.or_none()}
    rows = [
        (value,)
        for value in (*STRING_TRAPS, None, "NOISE", "Noİse", "xİ", "İx", "\u212a", "\u017f", "[A]")
    ]
    assert check_function(contains, schema, backend, rows) == []
    if needle in {"noise", "k", "s", "ix", "[a]"}:
        expr = backend.build(trace(contains, {"text": STR}))
        assert not fallback_in_plan(expr, backend.name)
        assert "lower(" not in str(expr).lower()


@pytest.mark.parametrize("pattern", ["(a)", "(?i)a", "(?P<letter>a)", "["])
def test_search_union_keeps_capture_and_flag_scope(pattern: str) -> None:
    text = Col("text", STR)
    first = apply("re.contains", text, lit(pattern), lit(0))
    second = apply("re.contains", text, lit("b"), lit(0))
    assert combined_search(apply("or", first, second)) is None
