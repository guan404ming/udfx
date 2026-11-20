from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from functools import partial, reduce

from pyspark.sql import Column, functions

from udfx.backends.engines.spark.rules import RULES
from udfx.frontend.core.text import CASED, STRIPPED, swapped
from udfx.semantics.text.characters import NONSPACE_ASCII, whitespace
from udfx.types import Kind

NESTED_LIMIT = 1000


def stripped(characters: Column | None) -> Column:
    space = functions.lit(whitespace())
    return space if characters is None else functions.coalesce(characters, space)


def nonspace(text: Column, side: int) -> Column:
    first, last = (functions.lit(bytes([byte])) for byte in NONSPACE_ASCII)
    return functions.substring(text.cast("binary"), side, 1).between(first, last)


def spaced(
    text: Column, spaces: Callable[[Column], Column], sides: Sequence[int], full: Column
) -> Column:
    if len(str(text)) > NESTED_LIMIT:
        return full
    trimmed = spaces(text)
    edges = reduce(operator.and_, (nonspace(trimmed, side) for side in sides))
    return functions.when(edges, trimmed).otherwise(full)


def strip(text: Column, characters: Column | None = None) -> Column:
    return functions.btrim(text, stripped(characters))


def lstrip(text: Column, characters: Column | None = None) -> Column:
    return functions.ltrim(text, stripped(characters))


def rstrip(text: Column, characters: Column | None = None) -> Column:
    return functions.rtrim(text, stripped(characters))


RULES(".strip", Kind.STR, Kind.STR)(strip)
RULES(".lstrip", Kind.STR, Kind.STR)(lstrip)
RULES(".rstrip", Kind.STR, Kind.STR)(rstrip)


@RULES(".strip", Kind.STR)
def strip_whitespace(text: Column) -> Column:
    return spaced(text, functions.trim, (1, -1), strip(text))


@RULES(".lstrip", Kind.STR)
def lstrip_whitespace(text: Column) -> Column:
    return spaced(text, functions.ltrim, (1,), lstrip(text))


@RULES(".rstrip", Kind.STR)
def rstrip_whitespace(text: Column) -> Column:
    return spaced(text, functions.rtrim, (-1,), rstrip(text))


for method in STRIPPED:
    RULES.rewrite(method, Kind.STR)(partial(swapped, names=CASED))
