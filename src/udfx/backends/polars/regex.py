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


def first(text: pl.Expr, pattern: str, group: int, nonempty: bool) -> pl.Expr:
    if nonempty:
        return text.str.extract(pattern, group)
    return (
        pl.when(text.str.contains(pattern))
        .then(text.str.extract(pattern, group).fill_null(""))
        .otherwise(None)
    )


register_regex(
    RULES,
    RegexPrimitives(
        dialect="rust",
        literal=literal,
        contains=lambda text, pattern: text.str.contains(pattern),
        extract=lambda text, pattern, empty: text.str.extract(pattern, 0),
        replace=replace,
        split=lambda text, separator: text.str.split(separator),
        findall=findall,
        first=first,
        count_matches=lambda text, pattern: text.str.count_matches(pattern).cast(pl.Int64),
        maximum_count=1,
    ),
)
