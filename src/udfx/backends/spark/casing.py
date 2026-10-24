from collections.abc import Callable

from pyspark.sql import Column, functions

from udfx.semantics.casing import case_mapping, final_sigma_pattern, simple_expansion
from udfx.semantics.text import SIGMA, char_class


def exact_case(
    value: Column,
    method: str,
    native: Column,
    exact: Column | None,
    characters: Callable[[Column, Callable[[Column], Column]], Column],
) -> Column:
    if exact is None:
        return native
    original, replacement, expansions = case_mapping(method)
    corrected = value
    if method == "lower":
        corrected = functions.when(
            value.contains(SIGMA), functions.regexp_replace(value, final_sigma_pattern(), "$1ς")
        ).otherwise(value)
    plain = functions.translate(corrected, original, replacement)
    expansion = simple_expansion(replacement, expansions)
    if expansion is not None:
        source, target = expansion
        plain = functions.replace(plain, functions.lit(source), functions.lit(target))
    elif expansions:
        table = functions.create_map(*(functions.lit(part) for pair in expansions for part in pair))
        expanded = characters(
            corrected,
            lambda point: functions.coalesce(
                functions.element_at(table, point),
                functions.translate(point, original, replacement),
            ),
        )
        plain = functions.when(
            corrected.rlike(char_class(source for source, _ in expansions)), expanded
        ).otherwise(plain)
    return functions.when(exact, native).otherwise(plain)
