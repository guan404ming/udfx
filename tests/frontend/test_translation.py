import string
from collections.abc import Callable
from typing import Any

import pytest
from conftest import frontend_backend_params

from udfx.backends.base import Backend
from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda text: text.translate(str.maketrans("", "", string.punctuation)),
        lambda text: text.translate(str.maketrans("ab", "ba")),
        lambda text: text.translate(str.maketrans({"a": "bc", "b": "a", "!": None})),
        lambda text: text.translate({97: "aa", 98: 97, 99: None}),
        lambda text: text.translate({97.0: "b", "b": "a", -1: "x"}),
        lambda text: text.translate(str.maketrans("😀é", "é😀", "!")),
        lambda text: text.translate({}),
        lambda text: text.translate(str.maketrans("aa", "bc", "a")),
    ],
)
def test_literal_translation(function: Callable[[str], str], backend: Backend[Any]) -> None:
    rows = [
        (text,)
        for text in (
            "",
            "abc!",
            "aabb",
            "Hello, world!",
            "😀é!",
            "\x00a",
            "\ue000ab\ue001",
            "$\\a",
            None,
        )
    ]
    assert check_function(function, {"text": STR.or_none()}, backend, rows) == []


def test_maketrans_errors_reach_handler() -> None:
    def function(text: str) -> str:
        try:
            return text.translate(str.maketrans("ab", "c"))
        except ValueError:
            return "invalid"

    assert check_function(function, {"text": STR}, PolarsBackend(), [("abc",)]) == []


def test_dynamic_maketrans_stays_unsupported() -> None:
    with pytest.raises(Unsupported, match="expected a constant"):
        trace(lambda text: text.translate(str.maketrans(text, "a")), {"text": STR})
