import re
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.semantics import regex
from udfx.semantics.regex_translation import Pattern
from udfx.semantics.regex_translation import translate as translate_pattern
from udfx.testing import check_function
from udfx.testing.text_traps import STRING_TRAPS
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("pattern", "flags"),
    [(r"\b\w+\b", 0), (r"\w+", re.IGNORECASE), (r"\b[\u3000\u2000]+", 0)],
)
def test_regex_uses_existing_ascii_guard(
    backend: Backend[Any], pattern: str, flags: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected: list[tuple[bool, str]] = []

    def translate(*arguments: Any, **keywords: Any) -> Pattern:
        output = translate_pattern(*arguments, **keywords)
        if arguments[0] == pattern:
            selected.append((bool(arguments[3]) if len(arguments) > 3 else False, output.source))
        return output

    def matches(text: str) -> list[str]:
        return re.findall(pattern, text, flags)

    monkeypatch.setattr(regex, "translate", translate)
    rows = [(value,) for value in STRING_TRAPS]
    assert check_function(matches, {"text": STR.or_none()}, backend, rows) == []
    if backend.name != "spark" or flags & re.IGNORECASE:
        restricted = [source for ascii_only, source in selected if ascii_only]
        assert restricted
        assert all(r"\x{660}" not in source for source in restricted)
