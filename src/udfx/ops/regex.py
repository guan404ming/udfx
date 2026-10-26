from __future__ import annotations

import re
from functools import partial

from udfx.ops.registry import Op, register
from udfx.types import BOOL, INT, STR, list_of


def match_text(name: str, text: str, pattern: str, flags: int) -> str | None:
    found = getattr(re, name)(pattern, text, flags)
    return None if found is None else found.group()


def contains(text: str, pattern: str, flags: int) -> bool:
    return re.search(pattern, text, flags) is not None


def substitute(text: str, pattern: str, flags: int, replacement: str, count: int) -> str:
    return re.sub(pattern, replacement, text, count=count, flags=flags)


def findall(text: str, pattern: str, flags: int) -> list[str]:
    return re.findall(pattern, text, flags)


def split(text: str, pattern: str, flags: int, count: int) -> list[str]:
    return re.split(pattern, text, maxsplit=count, flags=flags)


for name in ("search", "match", "fullmatch"):
    register(Op(f"re.{name}", (STR, STR, INT), STR.or_none(), partial(match_text, name)))
register(Op("re.contains", (STR, STR, INT), BOOL, contains))
register(Op("re.sub", (STR, STR, INT, STR, INT), STR, substitute))
register(Op("re.findall", (STR, STR, INT), list_of(STR), findall))
register(Op("re.split", (STR, STR, INT, INT), list_of(STR), split))
