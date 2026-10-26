from __future__ import annotations

import hashlib
import json
import operator
import os
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import cache, reduce
from pathlib import Path
from typing import Any

SIGMA = "\u03a3"
SIGMA_CONTEXTS = (f"A{SIGMA}{{}}b", f"A{{}}{SIGMA}", f"{{}}{SIGMA}", f"A{SIGMA}{{}}")

Recase = Callable[[Sequence[str]], Sequence[str]]
ANY_CHAR = "[\\s\\S]"
PREDICATES = (
    "isascii",
    "isdigit",
    "isalpha",
    "isalnum",
    "isnumeric",
    "isdecimal",
    "isspace",
    "islower",
    "isupper",
    "istitle",
)
CASE_METHODS = ("lower", "upper", "title", "capitalize", "swapcase", "casefold")
PADDING_METHODS = ("zfill", "center", "ljust", "rjust")
NATIVE_PADDING_LIMIT = 2**20


@cache
def code_points() -> tuple[str, ...]:
    return tuple(chr(point) for point in range(0x110000) if not 0xD800 <= point <= 0xDFFF)


@cache
def whitespace() -> str:
    return "".join(point for point in code_points() if point.isspace())


def char_class(chars: Iterable[str], negate: bool = False) -> str:
    ranges: list[str] = []
    points = sorted({ord(char) for char in chars})
    start = end = points[0] if points else 0
    for point in points[1:]:
        if point == end + 1:
            end = point
        else:
            ranges.append(f"\\x{{{start:X}}}" + (f"-\\x{{{end:X}}}" if start != end else ""))
            start = end = point
    if points:
        ranges.append(f"\\x{{{start:X}}}" + (f"-\\x{{{end:X}}}" if start != end else ""))
    return f"[{'^' if negate else ''}{''.join(ranges)}]"


@cache
def character_class(method: str) -> str:
    predicate: Callable[[str], bool] = getattr(str, method)
    return char_class(point for point in code_points() if predicate(point))


@cache
def cased_characters() -> str:
    return "".join(
        point for point in code_points() if point.islower() or point.isupper() or point.istitle()
    )


@cache
def predicate_patterns(method: str) -> tuple[str, str | None]:
    if method == "isascii":
        return r"\A[\x00-\x7f]*\z", None
    if method not in {"islower", "isupper", "istitle"}:
        return rf"\A{character_class(method)}+\z", None
    uncased = char_class(cased_characters(), negate=True)
    if method == "istitle":
        title = character_class("istitle")
        lowercase_start = rf"(?:\A|{uncased}){character_class('islower')}"
        uppercase_continuation = char_class(cased_characters()) + title
        return title, f"{lowercase_start}|{uppercase_continuation}"
    forbidden = char_class(
        (point for point in cased_characters() if not getattr(point, method)()),
        negate=True,
    )
    return rf"\A{uncased}*{character_class(method)}{forbidden}*\z", None


def string_predicate(matches: Callable[[Any, str], Any], value: Any, method: str) -> Any:
    required, forbidden = predicate_patterns(method)
    found = matches(value, required)
    return found if forbidden is None else found & ~matches(value, forbidden)


@cache
def whitespace_pattern(negate: bool = False, ascii_only: bool = False) -> str:
    characters = "".join(char for char in whitespace() if char.isascii() or not ascii_only)
    return f"[{'^' if negate else ''}{re.escape(characters)}]+"


@cache
def non_whitespace_pattern() -> str:
    return whitespace_pattern(negate=True)


@cache
def word_prefix_pattern() -> str:
    return char_class(whitespace()) + "*" + non_whitespace_pattern()


def strip_pattern(characters: str, method: str) -> str:
    pattern = char_class(characters) + "+"
    if method == "lstrip":
        return "^" + pattern
    if method == "rstrip":
        return pattern + "$"
    return "^" + pattern + "|" + pattern + "$"


def drift(reference: Callable[[str], str], native: Recase, context: str = "{}") -> frozenset[str]:
    points = code_points()
    samples = [context.format(point) for point in points]
    mapped = native(samples)
    return frozenset(
        point
        for point, sample, result in zip(points, samples, mapped, strict=True)
        if reference(sample) != result
    )


@dataclass(frozen=True)
class CaseDrift:
    alone: str | None
    nearby: str | None


def case_guard(
    found: CaseDrift, contains_class: Callable[[str], Any], contains_sigma: Any
) -> Any | None:
    if found.nearby == ANY_CHAR:
        pattern = f"(?:{found.alone}|{SIGMA})" if found.alone else SIGMA
        return ~contains_class(pattern)
    guards = [~contains_class(found.alone)] if found.alone else []
    if found.nearby:
        guards.append(~(contains_sigma & contains_class(found.nearby)))
    return reduce(operator.and_, guards) if guards else None


def case_drift(method: str, native: Recase, engine: str) -> CaseDrift:
    python = (
        f"python {sys.version_info[0]}.{sys.version_info[1]} unicode {unicodedata.unidata_version}"
    )
    key = f"{engine} {method} {python}"
    cached = read_cache(key)
    if cached is not None:
        return CaseDrift(**cached)
    reference: Callable[[str], str] = getattr(str, method)
    alone = drift(reference, native)
    if method in {"title", "capitalize"}:
        alone |= frozenset[str]().union(
            *(drift(reference, native, context) for context in ("A{}b", "a{}B", "{}b", "a{}"))
        )
    contexts = (drift(reference, native, context) for context in SIGMA_CONTEXTS)
    nearby = frozenset[str]().union(*contexts) - alone
    widespread = len(nearby) > len(code_points()) // 2
    found = CaseDrift(
        char_class(alone) if alone else None,
        ANY_CHAR if widespread else char_class(nearby) if nearby else None,
    )
    write_cache(key, asdict(found))
    return found


@dataclass(frozen=True)
class CasePrimitives:
    lower: Callable[[Any], Any]
    upper: Callable[[Any], Any]
    title: Callable[[Any], Any]
    head: Callable[[Any], Any]
    tail: Callable[[Any], Any]
    concat: Callable[..., Any]
    when: Callable[[Any, Any, Any], Any]
    matches: Callable[[Any, str], Any]
    characters: Callable[[Any, Callable[[Any], Any]], Any]


def capitalized(primitives: CasePrimitives, value: Any) -> Any:
    return primitives.concat(
        primitives.upper(primitives.head(value)), primitives.lower(primitives.tail(value))
    )


def case_expression(primitives: CasePrimitives, value: Any, method: str) -> Any:
    if method in {"lower", "casefold"}:
        return primitives.lower(value)
    if method == "upper":
        return primitives.upper(value)
    if method == "title":
        return primitives.title(value)
    if method == "capitalize":
        return capitalized(primitives, value)
    return primitives.characters(
        value,
        lambda character: primitives.when(
            primitives.matches(character, character_class("islower")),
            primitives.upper(character),
            primitives.lower(character),
        ),
    )


def cache_path(key: str) -> Path:
    root = os.environ.get("UDFX_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "udfx" / f"case-drift-{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def read_cache(key: str) -> dict[str, str | None] | None:
    try:
        return json.loads(cache_path(key).read_text())
    except (OSError, ValueError):
        return None


def write_cache(key: str, value: dict[str, str | None]) -> None:
    path = cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    except OSError:
        return


def has_prefix_suffix_overlap(pattern: str) -> bool:
    return any(pattern[:size] == pattern[-size:] for size in range(1, len(pattern)))
