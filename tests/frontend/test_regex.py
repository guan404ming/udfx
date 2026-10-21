from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import Call
from udfx.testing import check_function
from udfx.testing.strategies import STRING_TRAPS
from udfx.types import STR

COMPILED = re.compile(r"\d+")


@pytest.mark.parametrize(
    ("function", "name"),
    [
        (lambda text: re.sub(r"\d+", "#", text), "sub"),
        (lambda text: re.sub(pattern="a", repl="b", string=text, count=1, flags=re.I), "sub"),
        (lambda text: re.findall(r"(\d+)", text), "findall"),
        (lambda text: re.split(r"\s+", text, maxsplit=2), "split"),
        (lambda text: COMPILED.sub("#", text), "sub"),
        (lambda text: COMPILED.findall(string=text), "findall"),
        (lambda text: COMPILED.split(string=text), "split"),
        (lambda text: re.compile(r"\d+").sub("#", text, count=1), "sub"),
        (lambda text: re.sub(COMPILED, "#", text), "sub"),
    ],
)
@pytest.mark.parametrize("backend", backend_params())
def test_regex_call_forms(
    function: Callable[..., object], name: str, backend: Backend[Any]
) -> None:
    program = trace(function, {"text": STR})
    assert isinstance(program.value, Call)
    assert program.value.op.name == f"re.{name}"
    assert (
        check_function(function, {"text": STR}, backend, [(text,) for text in STRING_TRAPS]) == []
    )


@pytest.mark.parametrize("name", ["search", "match", "fullmatch"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("backend", backend_params())
def test_regex_match_tests(name: str, compiled: bool, backend: Backend[Any]) -> None:
    function = getattr(COMPILED if compiled else re, name)

    def tested(text: str) -> str:
        found = function(text) if compiled else function(r"\d+", text)
        if found:
            return found.group()
        return "missing"

    program = trace(tested, {"text": STR})
    assert program.result.strict() == STR
    assert check_function(tested, {"text": STR}, backend, [(text,) for text in STRING_TRAPS]) == []


@pytest.mark.parametrize(
    "function",
    [
        lambda text: re.search("x*", text) is None,
        lambda text: re.match("x*", text) is not None,
        lambda text: not re.fullmatch("x*", text),
        lambda text: bool(re.search("x*", text)),
        lambda text: re.compile("x*").search(text) is not None,
    ],
)
@pytest.mark.parametrize("backend", backend_params())
def test_regex_match_truth(function: Callable[..., object], backend: Backend[Any]) -> None:
    trace(function, {"text": STR})
    assert (
        check_function(function, {"text": STR}, backend, [(text,) for text in STRING_TRAPS]) == []
    )


def test_regex_none_guard() -> None:
    def first(text: str) -> str:
        found = re.search(r"\d+", text)
        if found is None:
            return ""
        return found.group(0)

    trace(first, {"text": STR})


@pytest.mark.parametrize("flag", [re.I, re.M, re.S, re.I | re.M | re.S])
@pytest.mark.parametrize("backend", backend_params())
def test_regex_flags(flag: int, backend: Backend[Any]) -> None:
    def substitute(text: str) -> str:
        return re.sub("a.*$", "", text, flags=flag)

    trace(substitute, {"text": STR})
    assert (
        check_function(substitute, {"text": STR}, backend, [(text,) for text in STRING_TRAPS]) == []
    )


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("pattern", ["x*", ".*?", r"\b\w+\b", r"[\u0130]"])
def test_regex_fallback(backend: Backend[Any], pattern: str) -> None:
    def substitute(text: str) -> str:
        return re.sub(pattern, "-", text, count=2, flags=re.I)

    assert (
        check_function(substitute, {"text": STR}, backend, [(text,) for text in STRING_TRAPS]) == []
    )


@pytest.mark.parametrize("backend", backend_params())
def test_regex_dynamic_replacement(backend: Backend[Any]) -> None:
    def substitute(text: str, replacement: str) -> str:
        return re.sub(r"\d+", replacement, text)

    assert (
        check_function(
            substitute,
            {"text": STR, "replacement": STR},
            backend,
            [("a12b", "#"), ("٣", r"\g<0>!")],
        )
        == []
    )


def test_regex_flag_expression() -> None:
    def substitute(text: str) -> str:
        return re.sub("a.*$", "", text, flags=0 | re.IGNORECASE | re.MULTILINE | re.DOTALL)

    trace(substitute, {"text": STR})


@pytest.mark.parametrize(
    ("function", "reason"),
    [
        (lambda text: re.sub(text, "", text), "pattern must be a literal"),
        (lambda text: re.sub("x", "", text, count=len(text)), "count must be a literal"),
        (lambda text: re.sub("x", "", text, flags=len(text)), "flags must be a literal"),
        (lambda text: re.split("x", text, maxsplit=len(text)), "maxsplit must be a literal"),
        (lambda text: re.sub("(?=a)", "", text), "assert"),
        (lambda text: re.sub("(?<=a)", "", text), "assert"),
        (lambda text: re.sub(r"(a)\1", "", text), "groupref"),
        (lambda text: re.sub("a++", "", text), "possessive"),
        (lambda text: re.sub("(?>a)", "", text), "atomic"),
        (lambda text: re.sub("(?P<name>a)", "", text), "named groups"),
        (lambda text: re.sub("a", "", text, flags=re.X), "flags"),
        (lambda text: re.sub("a", "", text, flags=re.A), "flags"),
        (lambda text: re.sub("a", "", text, flags=re.U), "flags"),
        (lambda text: re.sub("(?x)a", "", text), "flags"),
        (lambda text: re.findall("(a)(b)", text), "at most one"),
        (lambda text: re.split("(a)", text), "capture groups"),
        (lambda text: re.search("a", text), "regex match"),
        (lambda text: re.search("a", text).group(), "proves"),  # pyright: ignore[reportOptionalMemberAccess]
        (lambda text: COMPILED.search(text, 1), "pos"),
        (lambda text: re.sub("[", "", text), "invalid regex"),
        (lambda text: re.sub("a", r"\1", text), "replacement"),
    ],
)
def test_regex_unsupported(function: Callable[..., Any], reason: str) -> None:
    with pytest.raises(Unsupported, match=reason):
        trace(function, {"text": STR})


def test_regex_unsupported_group() -> None:
    def first(text: str) -> str:
        found = re.search("(a)", text)
        if found is not None:
            return found.group(1)
        return ""

    with pytest.raises(Unsupported, match="group"):
        trace(first, {"text": STR})
