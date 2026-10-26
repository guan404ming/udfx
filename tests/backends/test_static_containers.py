from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
def test_large_constant_mapping(backend: Backend[Any]) -> None:
    entries = {str(index): str(index + 1) for index in range(18)}
    namespace: dict[str, Any] = {}
    exec(
        f"def lookup(value):\n mapping = {entries!r}\n"
        " merged = {**mapping, '0': 'overwritten'}\n return merged.get(value, 'missing')",
        namespace,
    )
    rows = [(str(index),) for index in range(20)]
    assert check_function(namespace["lookup"], {"value": STR}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_constant_container_metadata(backend: Backend[Any]) -> None:
    def selected(text: str) -> bool:
        entries = (("suffix", []), ("prefix", ["a"]))
        return entries[0][1] == [] and text in frozenset(("x", "y"))

    assert check_function(selected, {"text": STR}, backend, [("x",), ("z",), ("",)]) == []
