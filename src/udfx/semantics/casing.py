from collections.abc import Sequence
from functools import cache

from udfx.semantics.text import SIGMA, char_class, code_points


def folded_case(methods: Sequence[str]) -> tuple[tuple[str, str], ...] | None:
    if methods[0] != "casefold":
        return None
    if all(method in {"lower", "casefold"} for method in methods):
        return ()
    if any(method in {"upper", "swapcase"} for method in methods):
        return casefold_replacements()
    return None


@cache
def casefold_replacements() -> tuple[tuple[str, str], ...]:
    return tuple(
        (point, folded)
        for point in code_points()
        if point.casefold() != (folded := point.upper().casefold())
    )


@cache
def case_mapping(method: str) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    reference = getattr(str, method)
    mapping = {point: reference(point) for point in code_points() if reference(point) != point}
    singles = {point: target for point, target in mapping.items() if len(target) == 1}
    expansions = tuple((point, target) for point, target in mapping.items() if len(target) != 1)
    return "".join(singles), "".join(singles.values()), expansions


def simple_expansion(
    replacement: str, expansions: tuple[tuple[str, str], ...]
) -> tuple[str, str] | None:
    if len(expansions) == 1 and expansions[0][0] not in replacement:
        return expansions[0]
    return None


@cache
def final_sigma_pattern() -> str:
    cased: set[str] = set()
    ignored: set[str] = set()
    for point in code_points():
        is_cased = point.islower() or point.isupper() or point.istitle()
        is_ignored = (
            (f"A{SIGMA}{point}0").lower()[1] == "ς"
            if is_cased
            else (f"A{point}{SIGMA}").lower()[-1] == "ς"
        )
        if is_ignored:
            ignored.add(point)
        elif is_cased:
            cased.add(point)
    letters, skipped = char_class(cased), char_class(ignored)
    return f"({letters}{skipped}*){SIGMA}(?!{skipped}*{letters})"
