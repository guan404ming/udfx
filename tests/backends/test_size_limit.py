from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.testing import check_function
from udfx.types import STR


def suffixed(text: str) -> str:
    return text + "x"


@pytest.mark.parametrize("backend", backend_params())
def test_size_limit_runs_python(backend: Backend[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    fallback = backend.fallback

    def counted(native: Any, exact: Any, program: Any) -> Any:
        calls.append(program)
        return fallback(native, exact, program)

    monkeypatch.setattr(backend, "fallback", counted)
    rows = [("a",), ("",)]
    assert check_function(suffixed, {"text": STR}, backend, rows) == []
    assert not calls
    monkeypatch.setattr(type(backend), "size_limit", 1)
    assert check_function(suffixed, {"text": STR}, backend, rows) == []
    assert calls
