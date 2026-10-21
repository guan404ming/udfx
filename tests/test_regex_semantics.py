import re

import pytest

from udfx.errors import Unsupported
from udfx.semantics.regex import category, replacement, translate


@pytest.mark.parametrize("dialect", ["rust", "re2", "java"])
def test_regex_translation(dialect: str) -> None:
    translated = translate(r"\A(?:ab|[\d_])+?(\s*)\Z", 0, dialect)
    assert translated.groups == 1
    assert not translated.empty
    assert translated.native
    assert r"\x{660}-\x{669}" in translated.source
    assert r"\x{1C}-\x{20}" in translated.source
    assert translated.source.endswith(r"\z")


@pytest.mark.parametrize(
    "pattern", [r"(?=a)", r"(?<=a)", r"(a)\1", "a++", "(?>a)", "(?P<name>a)", "(?#text)a"]
)
def test_regex_rejects_unsupported_syntax(pattern: str) -> None:
    with pytest.raises(Unsupported, match="regex"):
        translate(pattern, 0, "rust")


@pytest.mark.parametrize("flags", [re.ASCII, re.VERBOSE, re.LOCALE, re.DEBUG])
def test_regex_rejects_flags(flags: int) -> None:
    with pytest.raises(Unsupported, match="flags"):
        translate("a", flags, "rust")


def test_regex_guards() -> None:
    assert translate("x*", 0, "rust").empty
    assert translate(r"\bword\B", 0, "rust").boundary
    assert translate("a$", 0, "rust").dollar
    assert not translate("a$", re.MULTILINE, "rust").dollar
    assert not translate("a{1001}", 0, "re2").native
    assert not translate("İ", re.IGNORECASE, "rust").native
    assert r"\x{B2}" not in category("CATEGORY_DIGIT")


def test_regex_deep_pattern() -> None:
    with pytest.raises(Unsupported, match="invalid regex"):
        translate("(" * 1000 + "a" + ")" * 1000, 0, "rust")


@pytest.mark.parametrize("pattern", [r"\(\?#text\)", "[(?#text)]", "[](#?text)]", "[^](#?text)]"])
def test_regex_comment_characters(pattern: str) -> None:
    assert translate(pattern, 0, "rust").native


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [("rust", "${1}-$${1}-${1}"), ("re2", r"\1-${1}-\1"), ("java", r"$1-\${1}-$1")],
)
def test_regex_replacement(dialect: str, expected: str) -> None:
    assert replacement("(a)", 0, r"\1-${1}-\g<1>", dialect) == expected
