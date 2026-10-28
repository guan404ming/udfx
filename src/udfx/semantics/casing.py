import re
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any

from udfx.semantics.text import SIGMA, CasePrimitives, cased_characters, char_class, code_points


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
def lowercase_pattern(needle: str) -> str | None:
    if not needle or not needle.isascii() or needle.lower() != needle:
        return None
    original, replacement, expansions = case_mapping("lower")
    for _, target in expansions:
        if needle in target or any(
            needle.endswith(target[:size]) or needle.startswith(target[-size:])
            for size in range(1, len(target))
        ):
            return None
    sources: dict[str, str] = {}
    for point, target in zip(original, replacement, strict=True):
        sources[target] = sources.get(target, "") + point
    return "".join("[" + re.escape(point + sources.get(point, "")) + "]" for point in needle)


@cache
def initial_lowercase_values(values: tuple[str, ...]) -> tuple[str, ...]:
    original, replacement, expansions = case_mapping("lower")
    sources: dict[str, str] = {}
    for point, target in zip(original, replacement, strict=True):
        sources[target] = sources.get(target, "") + point
    alternatives: set[str] = set()
    for value in dict.fromkeys(values):
        if not value:
            continue
        first = value[0]
        unchanged = first if first.lower() == first else ""
        characters = unchanged + sources.get(first, "")
        alternatives.update(point + value[1:] for point in characters)
        alternatives.update(
            point + value[len(target) :] for point, target in expansions if value.startswith(target)
        )
    return tuple(sorted(alternatives))


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


def title_words(
    primitives: CasePrimitives,
    value: Any,
    transform: Callable[[Any, str, Callable[[Any], Any]], Any],
    *,
    initial: Callable[[Any], Any] | None = None,
    ascii_only: bool = False,
) -> Any:
    pattern = (
        "[A-Za-z]+|[^A-Za-z]+"
        if ascii_only
        else char_class(cased_characters()) + "+|" + char_class(cased_characters(), True) + "+"
    )
    first = primitives.upper if initial is None else initial
    return transform(
        value,
        pattern,
        lambda word: primitives.concat(
            first(primitives.head(word)), primitives.lower(primitives.tail(word))
        ),
    )
