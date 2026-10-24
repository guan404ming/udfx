from typing import Any

import pytest
from conftest import backend_params

from bench.speed.timing import fallback_in_plan
from udfx import compile
from udfx.backends.base import Backend
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("table", [{"ß": "ss", "ä": "ae", "a": "b", "!": None}, {"a": "aa"}])
def test_translation_dependencies_avoid_markers(
    backend: Backend[Any], table: dict[str, str | None]
) -> None:
    mapping = str.maketrans(table)

    def translate(text: str) -> str:
        return text.translate(mapping)

    schema = {"text": STR}
    translation = compile(translate, schema, backend)
    assert translation.ok, translation.unsupported
    assert not fallback_in_plan(translation.expr, backend.name)
    rows = [(text,) for text in ("", "abc", "äaß!", "😀äß", "\ue000\ue001äaß!")]
    assert check_function(translate, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_translation_cycles_keep_simultaneous_replacement(backend: Backend[Any]) -> None:
    mapping = str.maketrans({"a": "bb", "b": "aa"})

    def translate(text: str) -> str:
        return text.translate(mapping)

    rows = [(text,) for text in ("", "ab", "ba", "\ue000a\ue001b")]
    assert check_function(translate, {"text": STR}, backend, rows) == []
