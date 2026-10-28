from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ops import OpKey
from udfx.testing import check_function
from udfx.types import STR, Kind


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("separator", ["::", "aa", "[.]"])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_split_projection_preserves_bounds_without_arrays(
    backend: Backend[Any], separator: str, index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def selected(text: str) -> str | None:
        try:
            return text.split(separator)[index]
        except (AttributeError, IndexError):
            return None

    original = backend.rules.get

    def rule(key: OpKey) -> Any:
        if backend.name in ("datafusion", "spark"):
            assert key != OpKey(".split", (Kind.STR, Kind.STR))
        return original(key)

    monkeypatch.setattr(backend.rules, "get", rule)
    schema = {"text": STR.or_none()}
    values = [None, "", "a", "aaaaa", separator, separator * 3, f"\u03b1{separator}\n{separator}😀"]
    assert check_function(selected, schema, backend, [(value,) for value in values]) == []
    assert backend.lower_program(trace(selected, schema)).exact is None
