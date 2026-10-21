from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.ir import Col, lit
from udfx.ops import apply
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
def test_regex_native_execution(backend: Backend[Any]) -> None:
    text = Col("text", STR)
    for node in (
        apply("re.sub", text, lit(r"\d+"), lit(0), lit("#"), lit(0)),
        apply("re.findall", text, lit(r"\d+"), lit(0)),
        apply("re.split", text, lit(r"\d+"), lit(0), lit(0)),
        *(
            apply(f"re.{name}", text, lit(r"\d+"), lit(0))
            for name in ("search", "match", "fullmatch")
        ),
    ):
        lowered = backend.lower(node)
        assert lowered.exact is not None
        assert backend.evaluate([lowered.exact], {"text": STR}, [("12٣\U0001d7dc",), ("",)]) == [
            [True, True]
        ]
        assert str(lowered.expr).count("text") <= 3
