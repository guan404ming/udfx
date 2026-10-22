from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.polars import PolarsBackend
from udfx.frontend import trace
from udfx.ir import Guarded
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: value.title().swapcase().capitalize().casefold(),
        lambda value: value.upper().lower(),
        lambda value: value.lower().capitalize().casefold(),
    ],
)
def test_ascii_case_chain(backend: Backend[Any], function: Callable[[str], str]) -> None:
    schema = {"value": STR}
    program = trace(function, schema)
    assert isinstance(program.value, Guarded)
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
