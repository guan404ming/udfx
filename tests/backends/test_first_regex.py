import re
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ops import OpKey
from udfx.testing import check_function
from udfx.types import INT, STR, Kind

PATTERNS = (
    r"(?<!\d)[0-9]{6}(?!\d)",
    r"(?<![a-z])[ab]{2,3}(?![a-z])",
    r"(?<!_)_{2,4}(?!_)",
    r"(?<![^-])[^-]{2,4}(?![^-])",
)
VALUES = (
    None,
    "",
    "abc",
    "123456",
    "a123456b",
    "1234567 234567",
    "123456 234567",
    "\uff11\uff12\uff13\uff14\uff15\uff16",
    "१२3456",
    "१123456",
    "123456१",
    "\n123456\n",
    "123456\x00234567",
    "ab ba aaa aba",
    "cab ba",
    "_ __ ___ ____ _____",
    "-ab-a-abc-abcd-abcde-ab-",
)


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("pattern", PATTERNS)
def test_first_bounded_match_skips_run_filter(
    backend: Backend[Any], pattern: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def first(text: str) -> str:
        try:
            return re.findall(pattern, text)[0]
        except IndexError:
            return "missing"

    def forbidden(*arguments: object) -> Any:
        pytest.fail("Built an unused run array")

    original = backend.rules.get
    matches = OpKey("re.findall", (Kind.STR, Kind.STR, Kind.INT))
    monkeypatch.setattr(
        backend.rules, "get", lambda key: forbidden if key == matches else original(key)
    )
    schema = {"text": STR.or_none()}
    assert check_function(first, schema, backend, [(text,) for text in VALUES]) == []
    lowered = backend.lower_program(trace(first, schema))
    assert backend.evaluate(
        [backend.exact_expression(lowered)], schema, [(text,) for text in VALUES[1:]]
    ) == [[True] * (len(VALUES) - 1)]


@pytest.mark.parametrize("backend", backend_params())
def test_first_bounded_match_preserves_missing_item_error(backend: Backend[Any]) -> None:
    def first(text: str) -> str:
        return re.findall(r"(?<!\d)[0-9]{6}(?!\d)", text)[0]

    assert check_function(first, {"text": STR}, backend, [("none",), ("123456",)]) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda text: text if text is not None else "missing",
        lambda text: "missing" if text is None else text,
        lambda text: text if text != None else "missing",  # noqa: E711
    ],
)
def test_null_selection_uses_native_coalesce(
    backend: Backend[Any], function: Callable[..., str], monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = []
    original = backend.rules.get
    coalesce = OpKey("coalesce", (Kind.STR, Kind.STR))

    def get(key: OpKey) -> Any:
        if key == coalesce:
            observed.append(key)
        return original(key)

    monkeypatch.setattr(backend.rules, "get", get)
    assert (
        check_function(function, {"text": STR.or_none()}, backend, [(None,), ("",), ("a",)]) == []
    )
    assert observed


@pytest.mark.parametrize("backend", backend_params())
def test_null_selection_keeps_lazy_default_fault(backend: Backend[Any]) -> None:
    def selected(text: str | None, divisor: int) -> str:
        return text if text is not None else str(1 // divisor)

    rows = [("native", 0), (None, 1), (None, 0)]
    assert check_function(selected, {"text": STR.or_none(), "divisor": INT}, backend, rows) == []
