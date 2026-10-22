from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from udfx.errors import Unsupported
from udfx.semantics.regex_translation import Pattern, escaped, replacement, translate

SEPARATOR = "\x00"


@dataclass(frozen=True)
class RegexPrimitives:
    dialect: str
    literal: Callable[[Any], Any]
    contains: Callable[[Any, str], Any]
    extract: Callable[[Any, str], Any]
    replace: Callable[[Any, str, str, int], Any]
    split: Callable[[Any, str], Any]
    findall: Callable[[Any, str, int], Any]
    maximum_count: int


def lower_regex(
    primitives: RegexPrimitives,
    name: str,
    text: Any,
    pattern: Any,
    flags: Any,
    *,
    substitute: Any = None,
    count: Any = 0,
) -> tuple[Any, Any]:
    empty = (
        None
        if name in {"search", "match", "fullmatch"}
        else []
        if name in {"findall", "split"}
        else ""
    )
    fallback = primitives.literal(empty), primitives.literal(False)
    if not isinstance(pattern, str) or not isinstance(flags, int) or not isinstance(count, int):
        return fallback
    try:
        translated = translate(pattern, flags, primitives.dialect)
        if (
            not translated.native
            or (name in {"sub", "split", "findall"} and translated.empty)
            or (name == "findall" and translated.groups > 1)
            or (name == "split" and translated.groups)
            or count > primitives.maximum_count
        ):
            return fallback
        source = translated.source
        if name in {"match", "fullmatch"}:
            source = r"\A(?:" + source + ")" + (r"\z" if name == "fullmatch" else "")
        exact = regex_exact(primitives, translated, text)
        if name in {"search", "match", "fullmatch"}:
            return primitives.extract(text, source), exact
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


def regex_exact(primitives: RegexPrimitives, pattern: Pattern, text: Any) -> Any:
    exact = primitives.literal(True)
    if (pattern.boundary and primitives.dialect != "java") or pattern.ignorecase:
        exact = exact & ~primitives.contains(text, r"[^\x{0}-\x{7F}]")
    if pattern.boundary:
        exact = exact & (text != primitives.literal(""))
        if primitives.dialect == "java":
            exact = exact & ~primitives.contains(text, r"[\x{10000}-\x{10FFFF}]")
    if pattern.dollar:
        exact = exact & ~primitives.contains(text, r"\n")
    return exact
