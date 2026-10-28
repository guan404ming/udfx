import re
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR


def words(text: str) -> list[bool]:
    return [bool(re.search("a$", word)) for word in text.split()]


def limited(text: str) -> list[bool]:
    return [bool(re.search("a$", word)) for word in text.split(None, 1)]


def restored(text: str) -> list[bool]:
    return [bool(re.search("a$", re.sub("x", "a\\n", word))) for word in text.split()]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [words, limited, restored])
def test_regex_newline_facts(backend: Backend[Any], function: Callable[..., object]) -> None:
    schema = {"text": STR}
    rows = [(text,) for text in ("", "a", "a\n", "x\na\nx", "x a\n", "x\u2003a", "\na\nb\n")]
    assert check_function(function, schema, backend, rows) == []
    if function is words:
        assert backend.lower_program(trace(function, schema)).exact is None
