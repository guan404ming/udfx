import re
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("pattern", [r"[\d.]+\b", r"[\d.]+\B", r"[\w\s]+\b", r"[\w\s]+\B"])
def test_proven_boundary_keeps_native_unicode(backend: Backend[Any], pattern: str) -> None:
    def tokens(value: str) -> str:
        return "|".join(re.findall(pattern, value))

    rows = [
        (value,)
        for value in (
            "",
            "1.\U0001d7cea",
            "\U0001d7ce.\U0001d7cfa",
            "\U00010400 a\U0001d7ce",
            "a\x001.\U0001d7ce ",
            "1.2 ",
        )
    ]
    assert check_function(tokens, {"value": STR}, backend, rows) == []
    if backend.name == "spark":
        assert backend.lower_program(trace(tokens, {"value": STR})).exact is None
