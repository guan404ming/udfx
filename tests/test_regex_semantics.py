import re

import pytest

from udfx.errors import Unsupported
from udfx.semantics.regex import ascii_differs
from udfx.semantics.regex_categories import category
from udfx.semantics.regex_replacement import replacement
from udfx.semantics.regex_translation import translate


@pytest.mark.parametrize("dialect", ["rust", "re2", "java"])
def test_regex_translation(dialect: str) -> None:
    translated = translate(r"\A(?:ab|[\d_])+?(\s*)\Z", 0, dialect)
    assert translated.groups == 1
    assert not translated.empty
    assert translated.native
    assert r"\x{660}-\x{669}" in translated.source
    assert r"\x{1C}-\x{20}" in translated.source
    assert translated.source.endswith(r"\z")


@pytest.mark.parametrize("dialect", ["rust", "re2", "java", "datafusion"])
@pytest.mark.parametrize("pair", [r"\s\S", r"\d\D", r"\w\W"])
def test_regex_complementary_categories_stay_compact(dialect: str, pair: str) -> None:
    for prefix in ("", "^"):
        pattern = translate(f"[{prefix}{pair}]", 0, dialect)
        assert pattern.native
        assert len(pattern.source) < 20


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


@pytest.mark.parametrize(
    ("template", "expected"),
    [(r"\\\1", r"\${1}"), (r"\\u000\1", r"\u000${1}"), (r"\\", "\\")],
)
def test_datafusion_replacement_keeps_literal_backslashes(template: str, expected: str) -> None:
    assert replacement("(a)", 0, template, "datafusion") == expected


@pytest.mark.parametrize("template", [r"\\1", r"\\0", "\\\\\u0661"])
def test_datafusion_replacement_rejects_ambiguous_digits(template: str) -> None:
    with pytest.raises(Unsupported, match="before digits"):
        replacement("(a)", 0, template, "datafusion")


def test_regex_ascii_only_drops_wide_class_members() -> None:
    assert "3000" not in translate("[^ \\u3000]+\\Z", 0, "java", True).source
    assert translate("[\\u3000\\u2000]", 0, "java", True).source == "(?d)(?!)"
    assert "212A" in translate("(?i)[\\u212Ax]", 0, "java", True).source


@pytest.mark.parametrize("dialect", ["rust", "re2", "java", "datafusion"])
def test_ascii_guard_removes_redundant_regex_branch(dialect: str) -> None:
    assert not ascii_differs(r"\w+", re.IGNORECASE, dialect)
    assert ascii_differs(r"\b\w+\b", 0, dialect) is (dialect == "java")
    assert ascii_differs(r"\w+", 0, dialect)


def test_ascii_branch_keeps_patterns_above_unicode_expansion_limit() -> None:
    pattern = r"\w{100}"
    assert not translate(pattern, re.IGNORECASE, "java").native
    assert translate(pattern, re.IGNORECASE, "java", True).native
    assert ascii_differs(pattern, re.IGNORECASE, "java")


@pytest.mark.parametrize("pattern", ["a+", "[a-z]{2,4}", "[^a]*?", ".{1,3}", "(ab)+"])
def test_java_repetition_keeps_single_atoms(pattern: str) -> None:
    assert "(?:" not in translate(pattern, 0, "java").source


@pytest.mark.parametrize("pattern", ["(?:ab)+", "(?:a+)+"])
def test_java_repetition_groups_composite_atoms(pattern: str) -> None:
    assert "(?:" in translate(pattern, 0, "java").source


@pytest.mark.parametrize("pattern", [r"\w+\b", r"[\d.]+\b", r"[\w\s]+\B", r"(a)\b"])
def test_java_proven_suffix_removes_boundary_guard(pattern: str) -> None:
    assert not translate(pattern, 0, "java").boundary
    assert translate(pattern, 0, "rust").boundary


@pytest.mark.parametrize("pattern", [r"\w*\b", r"\b\w+", r"[\w\W]+\b", r"[^a]+\b", r"(?:)\b"])
def test_java_unknown_suffix_keeps_boundary_guard(pattern: str) -> None:
    assert translate(pattern, 0, "java").boundary


def test_java_case_insensitive_suffix_keeps_boundary_guard() -> None:
    assert translate(r"[\d.]+\b", re.IGNORECASE, "java").boundary


@pytest.mark.parametrize(
    "pattern", [r"(\d+)", "(a|b)", "(?i)(a+)", r"([\s\S]+)", "(a*)", "(?:((?:a|b)+))", "(a$)"]
)
def test_whole_capture_uses_the_complete_match(pattern: str) -> None:
    original = translate(pattern, 0, "rust")
    normalized = translate(pattern, 0, "rust", whole_capture=True)
    assert normalized.groups == 0
    assert normalized.empty == original.empty
    assert normalized.dollar == original.dollar
    assert normalized.ignorecase == original.ignorecase


@pytest.mark.parametrize(
    "pattern",
    [r"a(\d+)", "(a)?", "(a)+", "(a)|(b)", "((a))"],
)
def test_partial_capture_stays_unchanged(pattern: str) -> None:
    assert translate(pattern, 0, "rust", whole_capture=True) == translate(pattern, 0, "rust")
