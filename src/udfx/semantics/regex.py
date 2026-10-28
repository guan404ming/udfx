from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from udfx.errors import Unsupported
from udfx.semantics.regex_categories import escaped
from udfx.semantics.regex_replacement import replacement
from udfx.semantics.regex_translation import Pattern, translate
from udfx.semantics.text_predicates import AsciiPrimitives

SEPARATOR = "\x00"


@dataclass(frozen=True)
class RegexPrimitives:
    dialect: str
    literal: Callable[[Any], Any]
    contains: Callable[[Any, str], Any]
    extract: Callable[[Any, str, bool], Any]
    replace: Callable[[Any, str, str, int], Any]
    split: Callable[[Any, str], Any]
    findall: Callable[[Any, str, int], Any]
    first: Callable[[Any, str, int, bool], Any]
    count_matches: Callable[[Any, str], Any]
    maximum_count: int
    categories: Callable[[str], str] | None = None


def lower_regex(
    primitives: RegexPrimitives,
    name: str,
    text: Any,
    pattern: Any,
    flags: Any,
    *,
    substitute: Any = None,
    count: Any = 0,
    ascii: AsciiPrimitives | None = None,
    newline_free: bool = False,
) -> tuple[Any, Any]:
    options = {"substitute": substitute, "count": count, "newline_free": newline_free}
    expr, exact = regex_parts(primitives, name, text, pattern, flags, **options, ascii_only=False)
    if ascii is None or not ascii_differs(
        pattern, flags, primitives.dialect, primitives.categories
    ):
        return expr, exact
    fast, fast_exact = regex_parts(
        primitives, name, text, pattern, flags, **options, ascii_only=True
    )
    is_ascii = ascii.is_ascii(text)
    if fast_exact is not exact:
        exact = ascii.when(is_ascii, fast_exact, exact)
    return ascii.when(is_ascii, fast, expr), exact


def ascii_differs(
    pattern: Any, flags: Any, dialect: str, categories: Callable[[str], str] | None = None
) -> bool:
    if not isinstance(pattern, str) or not isinstance(flags, int):
        return False
    try:
        full = translate(pattern, flags, dialect, categories=categories)
        return (
            not (full.native and ascii_required(full, dialect))
            and full.source != translate(pattern, flags, dialect, True, categories).source
        )
    except Unsupported:
        return False


def regex_parts(
    primitives: RegexPrimitives,
    name: str,
    text: Any,
    pattern: Any,
    flags: Any,
    *,
    substitute: Any,
    count: Any,
    ascii_only: bool,
    newline_free: bool = False,
) -> tuple[Any, Any]:
    empty = (
        0
        if name == "count"
        else False
        if name == "contains"
        else None
        if name in {"search", "match", "fullmatch", "first"}
        else []
        if name in {"findall", "split"}
        else ""
    )
    fallback = primitives.literal(empty), primitives.literal(False)
    if not isinstance(pattern, str) or not isinstance(flags, int) or not isinstance(count, int):
        return fallback
    try:
        translated = translate(
            pattern, flags, primitives.dialect, ascii_only, primitives.categories
        )
        if (
            not translated.native
            or (name in {"sub", "split", "findall", "count"} and translated.empty)
            or (name in {"findall", "first"} and translated.groups > 1)
            or (name == "split" and translated.groups)
            or count > primitives.maximum_count
        ):
            return fallback
        source = (
            translate(pattern, flags, primitives.dialect, True, primitives.categories).source
            if not ascii_only and ascii_required(translated, primitives.dialect)
            else translated.source
        )
        if name in {"match", "fullmatch"}:
            source = r"\A(?:" + source + ")" + (r"\z" if name == "fullmatch" else "")
        exact = regex_exact(primitives, translated, text, newline_free)
        if name in {"contains", "count"}:
            measure = primitives.contains if name == "contains" else primitives.count_matches
            return measure(text, source), exact
        if name in {"search", "match", "fullmatch"}:
            return primitives.extract(text, source, translated.empty), exact
        if name == "first":
            nonempty = translated.capture_nonempty if translated.groups else not translated.empty
            return primitives.first(text, source, translated.groups, nonempty), exact
        if name == "findall":
            return primitives.findall(
                text, source, translated.groups
            ), exact & ~primitives.contains(text, escaped(0))
        if name == "split":
            if count < 0:
                return primitives.split(text, SEPARATOR), exact & ~primitives.contains(
                    text, escaped(0)
                )
            replaced = primitives.replace(text, source, SEPARATOR, count)
            return primitives.split(replaced, SEPARATOR), exact & ~primitives.contains(
                text, escaped(0)
            )
        if not isinstance(substitute, str):
            return fallback
        replaced = replacement(pattern, flags, substitute, primitives.dialect)
        if count < 0:
            return text, exact
        return primitives.replace(text, source, replaced, count), exact
    except Unsupported:
        return fallback


def ascii_required(pattern: Pattern, dialect: str) -> bool:
    return (pattern.boundary and dialect != "java") or pattern.ignorecase


def regex_exact(
    primitives: RegexPrimitives, pattern: Pattern, text: Any, newline_free: bool = False
) -> Any:
    exact: Any = True
    if ascii_required(pattern, primitives.dialect):
        exact = exact & ~primitives.contains(text, r"[^\x{0}-\x{7F}]")
    if pattern.boundary:
        exact = exact & (text != primitives.literal(""))
        if primitives.dialect == "java":
            exact = exact & ~primitives.contains(text, r"[\x{10000}-\x{10FFFF}]")
    if pattern.dollar and not newline_free:
        exact = exact & ~primitives.contains(text, r"\n")
    return exact
