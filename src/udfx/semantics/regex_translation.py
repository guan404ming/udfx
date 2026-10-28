from __future__ import annotations

import re
import re._constants as constants
import re._parser as parser
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

from udfx.errors import Unsupported
from udfx.semantics.regex_categories import category, escaped

FLAGS = int(re.IGNORECASE | re.MULTILINE | re.DOTALL | re.UNICODE)


def union_patterns(left: str, right: str, flags: int) -> str | None:
    try:
        patterns = [re.compile(pattern, flags) for pattern in (left, right)]
    except (re.error, ValueError, OverflowError, RecursionError):
        return None
    if any(pattern.groups or pattern.flags != flags | re.UNICODE for pattern in patterns):
        return None
    return f"(?:{left})|(?:{right})"


def ascii_member(name: str, member: Any) -> Any:
    if name == "LITERAL":
        return member if member <= 127 else None
    if name == "RANGE":
        return (member[0], min(member[1], 127)) if member[0] <= 127 else None
    return member


@dataclass(frozen=True)
class Pattern:
    source: str
    groups: int
    empty: bool
    boundary: bool
    dollar: bool
    ignorecase: bool
    native: bool
    capture_nonempty: bool


class Translator:
    def __init__(
        self,
        flags: int,
        dialect: str,
        ascii_only: bool,
        categories: Callable[[str], str] | None = None,
    ) -> None:
        self.categories = categories
        self.flags = flags
        self.dialect = dialect
        self.ascii_only = ascii_only
        self.boundary = False
        self.dollar = False
        self.native = True
        self.expansion = 0

    def sequence(self, tokens: Any, repetitions: int = 1) -> str:
        parts = [self.token(str(name), value, repetitions) for name, value in tokens]
        self.expansion += sum(map(len, parts)) * repetitions
        if self.expansion > (1_000_000 if self.dialect == "java" else 100_000):
            self.native = False
        return "".join(parts)

    def token(self, name: str, value: Any, repetitions: int) -> str:
        match name:
            case "LITERAL":
                if 0xD800 <= value <= 0xDFFF or (self.flags & re.IGNORECASE and value > 127):
                    self.native = False
                return escaped(value)
            case "NOT_LITERAL":
                return "[^" + self.token("LITERAL", value, repetitions) + "]"
            case "IN":
                return self.char_set(value)
            case "ANY":
                return "(?s:.)" if self.flags & re.DOTALL else "[^\\x{A}]"
            case "BRANCH":
                return "(?:" + "|".join(self.sequence(part, repetitions) for part in value[1]) + ")"
            case "SUBPATTERN":
                group, added, removed, body = value
                if added or removed:
                    raise Unsupported("regex scoped flags are unsupported")
                return ("(" if group else "(?:") + self.sequence(body, repetitions) + ")"
            case "MAX_REPEAT" | "MIN_REPEAT":
                minimum, maximum, body = value
                bound = minimum if maximum == constants.MAXREPEAT else maximum
                if repetitions * max(bound, 1) > 1000:
                    self.native = False
                suffix = "" if maximum == constants.MAXREPEAT else str(maximum)
                return (
                    "(?:"
                    + self.sequence(body, repetitions * max(bound, 1))
                    + ")"
                    + f"{{{minimum},{suffix}}}"
                    + ("?" if name == "MIN_REPEAT" else "")
                )
            case "AT":
                return self.anchor(str(value))
            case _:
                raise Unsupported(f"regex {name.lower()} is unsupported")

    def char_set(self, value: Any) -> str:
        negate = bool(value and str(value[0][0]) == "NEGATE")
        members = value[1:] if negate else value
        categories = {str(member) for name, member in members if str(name) == "CATEGORY"}
        if any("_NOT_" in name and name.replace("_NOT_", "_") in categories for name in categories):
            return r"[^\s\S]" if negate else "(?s:.)"
        if self.ascii_only and not self.flags & re.IGNORECASE:
            members = [
                (name, kept)
                for name, member in members
                if (kept := ascii_member(str(name), member)) is not None
            ]
            if not members:
                return "(?s:.)" if negate else "(?!)" if self.dialect == "java" else r"[^\s\S]"
        return (
            "["
            + ("^" if negate else "")
            + "".join(self.character(str(name), member) for name, member in members)
            + "]"
        )

    def character(self, name: str, value: Any) -> str:
        if name == "CATEGORY":
            return self.category(str(value))
        if name == "RANGE":
            if (value[0] <= 0xDFFF and value[1] >= 0xD800) or (
                self.flags & re.IGNORECASE and value[1] > 127
            ):
                self.native = False
            return escaped(value[0]) + "-" + escaped(value[1])
        return self.token(name, value, 1)

    def category(self, name: str) -> str:
        if self.categories is None or self.ascii_only:
            return category(name, self.ascii_only)
        return self.categories(name)

    def anchor(self, name: str) -> str:
        match name:
            case "AT_BEGINNING_STRING":
                return r"\A"
            case "AT_END_STRING":
                return r"\z"
            case "AT_BEGINNING":
                if self.dialect == "java" and self.flags & re.MULTILINE:
                    return r"(?:\A|(?<=\x{A}))"
                return "(?m:^)" if self.flags & re.MULTILINE else r"\A"
            case "AT_END":
                if self.dialect == "java":
                    return r"(?=\x{A}|\z)" if self.flags & re.MULTILINE else r"(?=\x{A}?\z)"
                if self.flags & re.MULTILINE:
                    return "(?m:$)"
                self.dollar = True
                return r"\z"
            case "AT_BOUNDARY" | "AT_NON_BOUNDARY":
                self.boundary = True
                if self.dialect == "java":
                    word = "[" + self.category("CATEGORY_WORD") + "]"
                    following, opposite = ("!", "=") if name == "AT_BOUNDARY" else ("=", "!")
                    return f"(?:(?<={word})(?{following}{word})|(?<!{word})(?{opposite}{word}))"
                return r"\b" if name == "AT_BOUNDARY" else r"\B"
            case _:
                raise Unsupported(f"regex anchor {name.lower()} is unsupported")


@cache
def translate(
    pattern: str,
    flags: int,
    dialect: str,
    ascii_only: bool = False,
    categories: Callable[[str], str] | None = None,
) -> Pattern:
    reject_comments(pattern)
    if flags & ~FLAGS:
        raise Unsupported("regex flags support only IGNORECASE, MULTILINE, and DOTALL")
    try:
        compiled = re.compile(pattern, flags)
        parsed = parser.parse(pattern, flags)
    except (re.error, ValueError, OverflowError, RecursionError) as error:
        raise Unsupported(f"invalid regex: {error}") from error
    if compiled.groupindex:
        raise Unsupported("regex named groups are unsupported")
    if parsed.state.flags & ~FLAGS:
        raise Unsupported("regex flags support only IGNORECASE, MULTILINE, and DOTALL")
    translator = Translator(parsed.state.flags, dialect, ascii_only, categories)
    source = translator.sequence(parsed)
    ignorecase = bool(parsed.state.flags & re.IGNORECASE)
    if ignorecase:
        source = "(?i:" + source + ")"
        if not pattern.isascii():
            translator.native = False
    if dialect == "java":
        source = "(?d)" + source
    return Pattern(
        source,
        compiled.groups,
        parsed.getwidth()[0] == 0,
        translator.boundary,
        translator.dollar,
        ignorecase,
        translator.native,
        any(
            str(name) == "SUBPATTERN" and value[0] == 1 and value[3].getwidth()[0] > 0
            for name, value in parsed
        ),
    )


def reject_comments(pattern: str) -> None:
    escaped = False
    class_start: int | None = None
    for index, point in enumerate(pattern):
        if escaped:
            escaped = False
        elif point == "\\":
            escaped = True
        elif class_start is not None:
            if point == "]" and index > class_start:
                class_start = None
        elif point == "[":
            class_start = index + 1 + pattern.startswith("^", index + 1)
        elif pattern.startswith("(?#", index):
            raise Unsupported("regex comments are unsupported")
