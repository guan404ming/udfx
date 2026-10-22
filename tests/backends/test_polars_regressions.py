from collections.abc import Callable
from typing import Any, Literal

import polars as pl
import pytest
from conftest import backend_params

from tests.corpus.cases_string_splitting import index_split_limit
from tests.corpus.cases_strings import join_characters
from udfx import compile
from udfx.backends.base import Backend
from udfx.testing import check_function
from udfx.types import INT, STR

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
def test_join_characters_on_empty_frames(engine: Literal["in-memory", "streaming"]) -> None:
    translation = compile(join_characters, {"text": STR, "separator": STR})
    assert translation.expr is not None, translation.unsupported
    frame = pl.DataFrame(schema={"text": pl.String, "separator": pl.String})
    assert frame.lazy().select(translation.expr).collect(engine=engine).to_series().to_list() == []
