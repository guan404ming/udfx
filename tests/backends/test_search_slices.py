import re
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Col, lit
from udfx.ops import apply
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


@pytest.mark.parametrize(
    "backend", [param for param in backend_params() if param.id == "datafusion"]
)
@pytest.mark.parametrize(
    ("pattern", "flags", "count", "rewritten"),
    [
        (r"-[\s\S]*\Z", 0, 0, True),
        (r"--[\s\S]*\Z", 0, 1, True),
        (r"\.[\s\S]*\Z", 0, 0, True),
        (r"-.*\Z", re.DOTALL, 0, True),
        (r"-.*\Z", 0, 0, False),
        (r"-[\s\S]*$", 0, 0, False),
        (r"-[\s\S]*\Z", 0, -1, False),
        (r"a[\s\S]*\Z", re.IGNORECASE, 0, False),
        (r"-[\s\S]+\Z", 0, 0, False),
        (r"[ab][\s\S]*\Z", 0, 0, False),
    ],
)
def test_suffix_removal_uses_literal_split(
    backend: Backend[Any], pattern: str, flags: int, count: int, rewritten: bool
) -> None:
    def selected(text: str) -> str:
        return re.sub(pattern, "", text, count=count, flags=flags)

    node = apply("re.sub", Col("text", STR), lit(pattern), lit(int(flags)), lit(""), lit(count))
    assert (backend.rules.rewrite_call(node) is not None) is rewritten
    samples = [None, "", "abc", "A尾", "-", "a--b--c", "a.b.c", "a-\n尾\x00\n", "a\n-b\nc"]
    assert check_function(selected, {"text": STR.or_none()}, backend, [(x,) for x in samples]) == []
