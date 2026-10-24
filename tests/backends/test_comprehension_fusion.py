from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR, list_of


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda items: "".join(str(index) + word for index, word in enumerate(items) if word != "a"),
        lambda items: "".join(word[1] for word in items if word),
        lambda items: "".join(word.casefold() for word in items if word != "a"),
    ],
)
def test_empty_separator_preserves_filtered_join(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    schema = {"items": list_of(STR).or_none()}
    rows = [([],), (["a"],), (["", "ab", "cd"],), (["a", "bb", "c"],), (["ß", "ɤ"],), (None,)]
    assert check_function(function, schema, backend, rows) == []


def test_spark_filtered_join_skips_pair_and_filter_arrays() -> None:
    def joined(items: list[str]) -> str:
        return "".join(str(index) + word for index, word in enumerate(items) if word != "a")

    expression = str(SparkBackend().build(trace(joined, {"items": list_of(STR)})))
    assert "filter(" not in expression
    assert expression.count("transform(") == 1
