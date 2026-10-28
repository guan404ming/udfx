from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"polars", "datafusion"}]
)
@pytest.mark.parametrize("separator", [" ", "|", "\n", "\\", "😀"])
@pytest.mark.parametrize(("limit", "inclusive"), [(1, True), (2, False)])
def test_joined_filter_skips_arrays(
    backend: Backend[Any], separator: str, limit: int, inclusive: bool
) -> None:
    def selected(text: str) -> str:
        try:
            if inclusive:
                return separator.join(word for word in text.split(separator) if len(word) >= limit)
            return separator.join(word for word in text.split(separator) if len(word) > limit)
        except AttributeError:
            return "missing"

    schema = {"text": STR.or_none()}
    rows = [
        (None,),
        ("",),
        (separator,),
        (separator * 5,),
        (separator.join(["a", "bb", "c"]),),
        (separator.join(["", "alpha", "b", "", "gamma", ""]),),
        (separator.join(["a\u0301", "é", "😀ab", "中文a"]),),
        (separator.join(["ab\x00", "ab\n", "ab\r", "ab\t", "ab\\"]),),
    ]
    with patch.object(backend, "list_filter", wraps=backend.list_filter) as filtered:
        assert check_function(selected, schema, backend, rows) == []
    filtered.assert_not_called()
    assert backend.lower_program(trace(selected, schema)).exact is None


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"polars", "datafusion"}]
)
def test_joined_filter_handles_large_literal_bound(backend: Backend[Any]) -> None:
    def selected(text: str) -> str:
        return " ".join(word for word in text.split(" ") if len(word) >= 64)

    rows = [("",), ("a" * 63 + " " + "b" * 64 + " " + "c" * 65,)]
    schema = {"text": STR}
    with patch.object(backend, "list_filter", wraps=backend.list_filter) as filtered:
        assert check_function(selected, schema, backend, rows) == []
    filtered.assert_not_called()
    assert backend.lower_program(trace(selected, schema)).exact is None


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id in {"polars", "datafusion"}]
)
@pytest.mark.parametrize(
    "function",
    [
        lambda text: "|".join(word for word in text.split(" ") if len(word) > 2),
        lambda text: " ".join(word for word in text.split(" ", 1) if len(word) > 2),
        lambda text: " ".join(word + "x" for word in text.split(" ") if len(word) > 2),
        lambda text: " ".join(word for word in text.split(" ") if len(word) == 2),
        lambda text: " ".join(word for word in text.split(" ") if len(word) >= 0),
        lambda text: " ".join(word for word in text.split(" ") if len(word) >= 65),
        lambda text: "::".join(word for word in text.split("::") if len(word) > 2),
    ],
)
def test_other_joined_filters_keep_arrays(
    backend: Backend[Any], function: Callable[[str], str]
) -> None:
    rows = [("",), ("  a bb ccc dddd ",), ("a::bb::ccc::::dddd",), ("é 😀 中文a",), ("x" * 65,)]
    with patch.object(backend, "list_filter", wraps=backend.list_filter) as filtered:
        assert check_function(function, {"text": STR}, backend, rows) == []
    assert filtered.called
