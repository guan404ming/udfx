import re
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import INT, STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("mode", ["search", "match", "fullmatch"])
@pytest.mark.parametrize("group", [0, 1])
def test_match_positions(backend: Backend[Any], mode: str, group: int) -> None:
    method = getattr(re.compile(r"a*(a|b)"), mode)

    def position(text: str, start: int) -> int:
        found = method(text, start)
        return found.start(group) if found else -1

    rows = [
        (text, start)
        for text in ("", "aaab", "zzaaab", "😀aab", "a\nb")
        for start in (-4, 0, 1, 20)
    ]
    assert check_function(position, {"text": STR, "start": INT}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_capture_position_with_empty_group(backend: Backend[Any]) -> None:
    pattern = re.compile(r"[^aeiouy]*[aeiouy]+[^aeiouy](\w*)")

    def region(text: str) -> int:
        found = pattern.match(text)
        return found.start(1) if found else len(text)

    rows = [(value,) for value in ("", "a", "ab", "aba", "abé", "ab😀", "bbb", "ends\n")]
    assert check_function(region, {"text": STR}, backend, rows) == []
    lowered = backend.lower_program(trace(region, {"text": STR}))
    flags = backend.evaluate([backend.exact_expression(lowered)], {"text": STR}, rows)[0]
    assert any(flag is True for flag in flags)


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("pattern", [r"(?<!\d)[0-9]{6}(?!\d)", r"(?<![a-z])[a-f]{2,4}(?![a-z])"])
def test_bounded_runs(backend: Backend[Any], pattern: str) -> None:
    def extract(text: str) -> list[str]:
        return re.findall(pattern, text)

    rows = [
        (value,)
        for value in (
            "",
            "a123456b654321c",
            "1234567 123456",
            "\u0661123456 123456\u0662",
            "123456\x00654321",
            "ab.cd.abcdef.abcd",
            "Zab cd!ef",
            "123456\n654321",
            "\U0001d7d9123456",
        )
    ]
    assert check_function(extract, {"text": STR}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("pattern", [r"(?i)a*(b)", r"(?m)^a*(b)"])
def test_position_keeps_inline_flags(backend: Backend[Any], pattern: str) -> None:
    def position(text: str) -> int:
        found = re.search(pattern, text)
        return found.start(1) if found else -1

    rows = [(value,) for value in ("AAAB", "xxaaab", "x\naaab", "İaab", "", "b")]
    assert check_function(position, {"text": STR}, backend, rows) == []
