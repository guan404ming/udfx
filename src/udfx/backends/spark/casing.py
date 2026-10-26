from collections.abc import Callable
from dataclasses import replace

from pyspark.sql import Column, functions

from udfx.backends.spark.rules import ASCII
from udfx.semantics.casing import case_mapping, final_sigma_pattern, simple_expansion, title_words
from udfx.semantics.text import SIGMA, CasePrimitives, char_class


def exact_case(
    value: Column,
    method: str,
    native: Column,
    exact: Column | None,
    characters: Callable[[Column, Callable[[Column], Column]], Column],
) -> Column:
    if exact is None:
        return native
    corrected = final_sigma(value) if method == "lower" else value
    return functions.when(exact, native).otherwise(mapped_case(corrected, method, characters))


def final_sigma(value: Column) -> Column:
    return functions.when(
        value.contains(SIGMA), functions.regexp_replace(value, final_sigma_pattern(), "$1ς")
    ).otherwise(value)


def mapped_point(value: Column, method: str) -> Column:
    original, replacement, expansions = case_mapping(method)
    plain = functions.translate(value, original, replacement)
    if not expansions:
        return plain
    table = functions.create_map(*(functions.lit(part) for pair in expansions for part in pair))
    return functions.coalesce(functions.element_at(table, value), plain)


def mapped_case(
    value: Column, method: str, characters: Callable[[Column, Callable[[Column], Column]], Column]
) -> Column:
    original, replacement, expansions = case_mapping(method)
    plain = functions.translate(value, original, replacement)
    expansion = simple_expansion(replacement, expansions)
    if expansion is not None:
        source, target = expansion
        plain = functions.replace(plain, functions.lit(source), functions.lit(target))
    elif expansions:
        expanded = characters(value, lambda point: mapped_point(point, method))
        plain = functions.when(
            value.rlike(char_class(source for source, _ in expansions)), expanded
        ).otherwise(plain)
    return plain


def exact_title(value: Column, primitives: CasePrimitives) -> Column:
    def words(value: Column, pattern: str, transform: Callable[[Column], Column]) -> Column:
        parts = functions.regexp_extract_all(value, functions.lit(pattern), 0)
        return functions.array_join(functions.transform(parts, transform), "")

    mapped = replace(
        primitives, lower=lambda value: mapped_case(value, "lower", primitives.characters)
    )
    full = title_words(
        mapped,
        final_sigma(value),
        words,
        initial=lambda point: mapped_point(point, "title"),
    )
    fast = title_words(primitives, value, words, ascii_only=True)
    return functions.when(ASCII.is_ascii(value), fast).otherwise(full)
