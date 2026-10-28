from __future__ import annotations

import re
import re._constants as constants
import re._parser as parser
from functools import cache
from typing import Any

from udfx.errors import Unsupported
from udfx.semantics.regex_translation import Translator
from udfx.semantics.text import code_points


def literal_union(values: tuple[str, ...]) -> str | None:
    groups: dict[str, set[str]] = {}
    for value in values:
        if value:
            groups.setdefault(value[1:], set()).add(value[0])
    alternatives = [""] if "" in values else []
    for suffix, first in sorted(groups.items()):
        characters = re.escape("".join(sorted(first)))
        prefix = "[" + characters + "]" if len(first) > 1 else characters
        alternatives.append(prefix + re.escape(suffix))
    return "(?:" + "|".join(alternatives) + ")" if alternatives else None


def source(tokens: Any, flags: int) -> str:
    translated = Translator(flags, "rust", False, python_category).sequence(tokens)
    return re.sub(
        r"\\x\{([0-9A-F]+)\}", lambda match: f"\\U{int(match[1], 16):08x}", translated
    ).replace(r"\z", r"\Z")


def python_category(name: str) -> str:
    category = {"DIGIT": "d", "SPACE": "s", "WORD": "w"}[name.rsplit("_", 1)[1]]
    return "\\" + (category.upper() if "_NOT_" in name else category)


def group_prefix(pattern: str, flags: int, group: int, mode: str) -> str:
    parsed: Any = parser.parse(pattern, flags)
    flags = parsed.state.flags
    if not 0 <= group < parsed.state.groups:
        raise Unsupported("regex group index is outside the pattern")
    position = 0
    if group:
        for index, (name, value) in enumerate(parsed):
            if str(name) == "SUBPATTERN" and value[0] == group:
                position = index
                break
        else:
            raise Unsupported("regex positions require a top-level capture group")
    prefix = (r"[\s\S]*?" if mode == "re.search" else "") + source(parsed[:position], flags)
    suffix = r"\Z" if mode == "re.fullmatch" else r"[\s\S]*\Z"
    enabled = "".join(
        name for flag, name in ((re.I, "i"), (re.M, "m"), (re.S, "s")) if flags & flag
    )
    options = f"(?{enabled})" if enabled else ""
    return options + r"\A(" + prefix + ")" + source(parsed[position:], flags) + suffix


def position_safe(pattern: str, flags: int) -> bool:
    pending = list(parser.parse(pattern, flags).data)
    while pending:
        name, value = pending.pop()
        if str(name) in {"AT", "ASSERT", "ASSERT_NOT"}:
            return False
        if str(name) == "SUBPATTERN":
            pending.extend(value[3])
        elif str(name) in {"MAX_REPEAT", "MIN_REPEAT"}:
            pending.extend(value[2])
        elif str(name) == "BRANCH":
            for branch in value[1]:
                pending.extend(branch)
    return True


def literal_suffix(pattern: str, flags: int) -> str | None:
    try:
        parsed: Any = parser.parse(pattern, flags)
    except (re.error, ValueError, OverflowError, RecursionError):
        return None
    if parsed.state.flags & ~(re.UNICODE | re.DOTALL) or len(parsed) < 3:
        return None
    *prefix, (name, repeated), ending = parsed
    if (
        any(str(name) != "LITERAL" or 0xD800 <= value <= 0xDFFF for name, value in prefix)
        or str(name) != "MAX_REPEAT"
        or repeated[:2] != (0, constants.MAXREPEAT)
        or len(repeated[2]) != 1
        or tuple(map(str, ending)) != ("AT", "AT_END_STRING")
    ):
        return None
    name, body = repeated[2][0]
    complete = str(name) == "ANY" and parsed.state.flags & re.DOTALL
    if str(name) == "IN":
        complete = {(str(name), str(value)) for name, value in body} == {
            ("CATEGORY", "CATEGORY_SPACE"),
            ("CATEGORY", "CATEGORY_NOT_SPACE"),
        }
    return "".join(chr(value) for _, value in prefix) if complete else None


@cache
def bounded_runs(pattern: str, flags: int) -> tuple[str, str] | None:
    if flags & ~re.UNICODE:
        return None
    try:
        parsed: Any = parser.parse(pattern, flags)
    except (re.error, ValueError, OverflowError, RecursionError):
        return None
    if parsed.state.flags & ~re.UNICODE or len(parsed) != 3 or parsed.state.groups != 1:
        return None
    (left_name, left), (body_name, body), (right_name, right) = parsed
    if (
        str(left_name) != "ASSERT_NOT"
        or str(right_name) != "ASSERT_NOT"
        or left[0] != -1
        or right[0] != 1
        or len(left[1]) != 1
        or list(left[1]) != list(right[1])
        or str(left[1][0][0]) not in {"LITERAL", "IN", "NOT_LITERAL"}
        or str(body_name) not in {"MAX_REPEAT", "MIN_REPEAT"}
        or body[0] < 1
        or len(body[2]) != 1
        or str(body[2][0][0]) not in {"LITERAL", "IN", "NOT_LITERAL"}
    ):
        return None
    boundary = source(left[1], flags)
    member = re.compile(source(body[2], flags))
    excluded = re.compile(boundary)
    if any(member.fullmatch(point) and not excluded.fullmatch(point) for point in code_points()):
        return None
    return f"(?:{boundary})+", source([(body_name, body)], flags)


@cache
def first_run(run: str, member: str, flags: int) -> str | None:
    try:
        parsed: Any = parser.parse(run, flags)
    except (re.error, ValueError, OverflowError, RecursionError):
        return None
    if len(parsed) != 1 or str(parsed[0][0]) != "MAX_REPEAT":
        return None
    minimum, maximum, repeated = parsed[0][1]
    if minimum != 1 or maximum != constants.MAXREPEAT or len(repeated) != 1:
        return None
    name, body = repeated[0]
    if str(name) == "IN":
        inverted = body[1:] if str(body[0][0]) == "NEGATE" else [("NEGATE", None), *body]
        outside = source([("IN", inverted)], flags)
    elif str(name) in {"LITERAL", "NOT_LITERAL"}:
        opposite = "NOT_LITERAL" if str(name) == "LITERAL" else "LITERAL"
        outside = source([(opposite, body)], flags)
    else:
        return None
    boundary = source(repeated, flags)
    if bounded_runs(f"(?<!{boundary}){member}(?!{boundary})", flags) is None:
        return None
    return rf"(?:\A|{outside})({member})(?:{outside}|\Z)"
