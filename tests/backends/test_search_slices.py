from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.testing.text_traps import STRING_TRAPS
from udfx.types import STR


def prefix(text: str) -> str:
    return text[: text.find("-")]


def middle(text: str) -> str:
    return text[text.find("-") + 1 : text.rfind("-")]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [prefix, middle])
def test_search_slices_preserve_missing_and_repeated_separators(
    backend: Backend[Any], function: Callable[[str], str]
) -> None:
    samples = [
        *STRING_TRAPS,
        "-",
        "--",
        "---",
        "first-last",
        "first-middle-last",
        "-middle-",
        "first-\n中\x00-\nlast",
        "first-😀-last",
        "a-b-c-d",
    ]
    assert check_function(function, {"text": STR}, backend, [(value,) for value in samples]) == []
    expression = str(backend.build(trace(function, {"text": STR})))
    assert len(expression) < 4000


@pytest.mark.parametrize("backend", backend_params())
def test_search_slice_conversion_preserves_large_integers(backend: Backend[Any]) -> None:
    def convert(text: str) -> int:
        try:
            return int(text[text.find("-") + 1 : text.rfind("-")])
        except ValueError:
            return -1

    values = ["", "2024-01-01", "a-٣_٢-b", "a-9223372036854775808-b", "a- 12 -b", "a-1__2-b"]
    assert check_function(convert, {"text": STR}, backend, [(value,) for value in values]) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("separator", ["\\", "]", "\n", "😀"])
@pytest.mark.parametrize("first", [True, False])
def test_search_slice_escapes_literal_separator(
    backend: Backend[Any], separator: str, first: bool
) -> None:
    def selected(text: str) -> str:
        if first:
            return text[: text.find(separator)]
        return text[text.find(separator) + 1 : text.rfind(separator)]

    samples = ["", "without separator", separator, separator * 3, separator.join(["a", "b", "c"])]
    assert check_function(selected, {"text": STR}, backend, [(value,) for value in samples]) == []
