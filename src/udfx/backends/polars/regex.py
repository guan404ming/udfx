from __future__ import annotations

import polars as pl

from udfx.backends.polars.rules import RULES
from udfx.backends.registration import register_regex
from udfx.semantics.regex import SEPARATOR, RegexPrimitives


def literal(value: object) -> pl.Expr:
    return pl.lit(value, dtype=pl.List(pl.String) if isinstance(value, list) else None)


def replace(text: pl.Expr, pattern: str, replacement: str, count: int) -> pl.Expr:
    return (
        text.str.replace(pattern, replacement, n=count)
        if count
        else text.str.replace_all(pattern, replacement)
    )


def findall(text: pl.Expr, pattern: str, group: int) -> pl.Expr:
    return (
        text.str.replace_all(pattern, f"{SEPARATOR}${{{group}}}{SEPARATOR}")
        .str.split(SEPARATOR)
        .list.slice(1)
        .list.gather_every(2)
    )


register_regex(
    RULES,
    RegexPrimitives(
        dialect="rust",
        literal=literal,
        contains=lambda text, pattern: text.str.contains(pattern),
        extract=lambda text, pattern: text.str.extract(pattern, 0),
        replace=replace,
        split=lambda text, separator: text.str.split(separator),
        findall=findall,
        maximum_count=1,
    ),
)
