from __future__ import annotations

import re
import re._constants as constants
import re._parser as parser
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from typing import Any

from udfx.errors import Unsupported
from udfx.semantics.text import code_points

FLAGS = int(re.IGNORECASE | re.MULTILINE | re.DOTALL | re.UNICODE)


def escaped(point: int) -> str:
    return f"\\x{{{point:X}}}"


def ranges(points: Iterable[int]) -> str:
    parts: list[str] = []
    start = previous = -1
    for point in points:
        if start >= 0 and point == previous + 1:
            previous = point
            continue
        if start >= 0:
            parts.append(escaped(start) + ("-" + escaped(previous) if start != previous else ""))
        start = previous = point
    if start >= 0:
        parts.append(escaped(start) + ("-" + escaped(previous) if start != previous else ""))
    return "".join(parts)


@cache
def category(name: str) -> str:
    positive = name.replace("_NOT_", "_")
    predicate = {
        "CATEGORY_DIGIT": str.isdecimal,
        "CATEGORY_SPACE": str.isspace,
        "CATEGORY_WORD": lambda point: point.isalnum() or point == "_",
    }[positive]
    negate = "_NOT_" in name
    return ranges(ord(point) for point in code_points() if predicate(point) != negate)


@dataclass(frozen=True)
class Pattern:
    source: str
    groups: int
    empty: bool
    boundary: bool
    dollar: bool
    ignorecase: bool
    native: bool


class Translator:
    def __init__(self, flags: int, dialect: str) -> None:
        self.flags = flags
        self.dialect = dialect
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
                negate = bool(value and str(value[0][0]) == "NEGATE")
                members = value[1:] if negate else value
                return (
                    "["
                    + ("^" if negate else "")
                    + "".join(self.character(str(name), member) for name, member in members)
                    + "]"
                )
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

    def character(self, name: str, value: Any) -> str:
        if name == "CATEGORY":
            return category(str(value))
        if name == "RANGE":
            if (value[0] <= 0xDFFF and value[1] >= 0xD800) or (
                self.flags & re.IGNORECASE and value[1] > 127
            ):
                self.native = False
            return escaped(value[0]) + "-" + escaped(value[1])
        return self.token(name, value, 1)

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
                    word = "[" + category("CATEGORY_WORD") + "]"
                    following, opposite = ("!", "=") if name == "AT_BOUNDARY" else ("=", "!")
                    return f"(?:(?<={word})(?{following}{word})|(?<!{word})(?{opposite}{word}))"
                return r"\b" if name == "AT_BOUNDARY" else r"\B"
            case _:
                raise Unsupported(f"regex anchor {name.lower()} is unsupported")


@cache
def translate(pattern: str, flags: int, dialect: str) -> Pattern:
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
    translator = Translator(parsed.state.flags, dialect)
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


def replacement(pattern: str, flags: int, value: str, dialect: str) -> str:
    if any(0xD800 <= ord(point) <= 0xDFFF for point in value):
        raise Unsupported("regex replacement surrogates need fallback")
    try:
        parts = parser.parse_template(value, re.compile(pattern, flags))
    except (re.error, IndexError) as error:
        raise Unsupported(f"invalid regex replacement: {error}") from error
    result: list[str] = []
    for index, part in enumerate(parts):
        if isinstance(part, int):
            if dialect == "re2":
                if part > 9:
                    raise Unsupported("RE2 replacement supports groups 0 through 9")
                result.append(f"\\{part}")
            else:
                result.append(f"${part}" if dialect == "java" else f"${{{part}}}")
        elif dialect in {"rust", "datafusion"}:
            if dialect == "datafusion" and "\\" in part:
                raise Unsupported("DataFusion literal replacement backslashes need fallback")
            result.append(part.replace("$", "$$"))
        else:
            escaped_part = part.replace("\\", "\\\\")
            if (
                dialect == "java"
                and index
                and isinstance(parts[index - 1], int)
                and part
                and part[0] in "0123456789"
            ):
                escaped_part = "\\" + escaped_part
            result.append(escaped_part.replace("$", r"\$") if dialect == "java" else escaped_part)
    return "".join(result)
