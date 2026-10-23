from collections.abc import Callable
from typing import Any, Literal

import polars as pl
import pytest
from conftest import backend_params

from tests.corpus.cases_lists import unpack_ends, weighted_word_length
from tests.corpus.cases_string_splitting import index_split_limit
from tests.corpus.cases_strings import join_characters
from tests.corpus.cases_structure import remove_stopwords
from udfx import compile
from udfx.backends.base import Backend
from udfx.testing import check_function
from udfx.types import INT, STR, Type

SCHEMA = {"text": STR, "maximum": INT, "index": INT}
ROWS = [("", 0, 1), ("1_1", 1, 0)]


def guarded_index_split_limit(text: str, maximum: int, index: int) -> str | None:
    return text.split("_", maximum)[index] if index < len(text.split("_", maximum)) else None


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [index_split_limit, guarded_index_split_limit])
def test_split_index_empty_branch(backend: Backend[Any], function: Callable[..., object]) -> None:
    assert check_function(function, SCHEMA, backend, ROWS) == []


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("function", [index_split_limit, guarded_index_split_limit])
def test_split_index_empty_branch_lazy(
    function: Callable[..., object], engine: Literal["in-memory", "streaming"]
) -> None:
    translation = compile(function, SCHEMA)
    assert translation.expr is not None, translation.unsupported
    frame = pl.DataFrame(ROWS, schema=["text", "maximum", "index"], orient="row")
    actual = frame.lazy().select(translation.expr).collect(engine=engine).to_series().to_list()
    assert actual == [function(*row) for row in ROWS]


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    ("function", "schema"),
    [(join_characters, {"text": STR, "separator": STR}), (remove_stopwords, {"text": STR})],
)
def test_joins_on_empty_frames(
    engine: Literal["in-memory", "streaming"],
    function: Callable[..., object],
    schema: dict[str, Type],
) -> None:
    translation = compile(function, schema)
    assert translation.expr is not None, translation.unsupported
    frame = pl.DataFrame(schema=dict.fromkeys(schema, pl.String))
    assert frame.lazy().select(translation.expr).collect(engine=engine).to_series().to_list() == []


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_fallback_on_chunked_frames(engine: Literal["in-memory", "streaming"]) -> None:
    translation = compile(weighted_word_length, {"text": STR.or_none()})
    assert translation.expr is not None, translation.unsupported
    texts = ["a bb"] + ["ccc"] * 2000
    frame = pl.concat([pl.DataFrame({"text": texts[:1]}), pl.DataFrame({"text": texts[1:]})])
    assert frame.n_chunks() == 2
    result = frame.lazy().select(translation.expr).collect(engine=engine).to_series().to_list()
    assert result == [weighted_word_length(text) for text in texts]


def test_word_ends_match_without_split() -> None:
    translation = compile(unpack_ends, {"text": STR.or_none()})
    assert translation.expr is not None, translation.unsupported
    assert "split" not in str(translation.expr)
