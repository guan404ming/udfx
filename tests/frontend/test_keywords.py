# pyright: reportCallIssue=false
# ruff: noqa: UP032
import sys
from collections.abc import Callable

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import Col, Lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import FLOAT, STR, Type

from .test_helpers import clean_text, keyword


def keyword_helper(text: str, /, prefix: str = "[", *, suffix: str = "]") -> str:
    return prefix + text + suffix


def nested_keyword_helper(text: str) -> str:
    def helper(value: str, *, suffix: str = "!") -> str:
        return value + suffix

    return helper(value=text)


@pytest.mark.parametrize(
    ("function", "schema", "rows"),
    [
        (keyword, {"s": STR}, [("a",), ("İ",)]),
        (lambda text: clean_text(suffix="?", text=text), {"text": STR}, [(" A ",)]),
        (lambda text: keyword_helper(text, suffix="!"), {"text": STR}, [("a",)]),
        (nested_keyword_helper, {"text": STR}, [("a",)]),
        (lambda text: text.split(sep=","), {"text": STR}, [("a,b",), ("",)]),
        (lambda value: round(value, ndigits=2), {"value": FLOAT}, [(2.675,), (1.125,)]),
        (lambda value: round(ndigits=2, number=value), {"value": FLOAT}, [(2.675,)]),
        (lambda value: round(number=value), {"value": FLOAT}, [(2.675,)]),
    ],
)
def test_keywords_match_python(
    function: Callable[..., object], schema: dict[str, Type], rows: list[tuple[object, ...]]
) -> None:
    assert check_function(function, schema, PolarsBackend(), rows) == []


def test_keyword_state_does_not_leak_to_next_call() -> None:
    def function(text: str) -> str:
        result = clean_text(text=text)
        return result.upper()

    assert trace(function, {"text": STR}).value == apply(
        ".upper",
        apply("__add__", apply(".lower", apply(".strip", Col("text", STR))), Lit("!", STR)),
    )


@pytest.mark.parametrize(
    ("function", "schema", "reason"),
    [
        (lambda text: clean_text(text, text=text), {"text": STR}, "multiple values"),
        (lambda text: clean_text(value=text), {"text": STR}, "missing a required argument"),
        (lambda text: clean_text(text, unknown="!"), {"text": STR}, "unexpected keyword"),
        (lambda text: keyword_helper(text=text), {"text": STR}, "positional.only"),
        (lambda text: text.split(",", sep=","), {"text": STR}, "multiple values"),
        (lambda text: text.split(unknown=","), {"text": STR}, "unexpected keyword"),
        (lambda text: text.strip(chars=" "), {"text": STR}, "positional.only"),
        (lambda text: int(text, base=10), {"text": STR}, "keyword arguments.*no signature"),
        (lambda text: "{text}".format(text=text), {"text": STR}, "keyword arguments"),
        (lambda text: clean_text(*(text,)), {"text": STR}, "unpacked call arguments"),
        (lambda text: clean_text(text, **{}), {"text": STR}, "unpacked call arguments"),
    ],
)
def test_unsupported_keywords(
    function: Callable[..., object], schema: dict[str, Type], reason: str
) -> None:
    with pytest.raises(Unsupported, match=reason):
        trace(function, schema)


@pytest.mark.parametrize("keyword", [False, True])
def test_variable_helper_parameters_remain_unsupported(keyword: bool) -> None:
    def positional(*values: str) -> str:
        return values[0]

    def keywords(**values: str) -> str:
        return values["text"]

    def function(text: str) -> str:
        return keywords(text=text) if keyword else positional(text)

    with pytest.raises(Unsupported, match="variable parameters"):
        trace(function, {"text": STR})


def test_replace_count_respects_the_python_signature() -> None:
    reason = "positional.only" if sys.version_info < (3, 13) else "no op .replace"
    with pytest.raises(Unsupported, match=reason):
        trace(lambda text: text.replace("a", "b", count=1), {"text": STR})


def test_keyword_binding_fills_a_skipped_default() -> None:
    with pytest.raises(Unsupported, match=r"no op .split\(str, none, int\)"):
        trace(lambda text: text.split(maxsplit=1), {"text": STR})
