from udfx.types import BOOL, FLOAT, INT, STR

from . import Case


def classify_text(value: str) -> str:
    if value.isdecimal():
        return "decimal"
    if value.isdigit():
        return "digit"
    if value.isnumeric():
        return "numeric"
    if value.isspace():
        return "space"
    if value.isalpha() and (value.islower() or value.isupper() or value.istitle()):
        return "letters"
    return "alphanumeric" if value.isalnum() else "other"


def local_membership(value: int) -> bool:
    choices = [1, 2, 3, 4]
    return value in choices


CASES = [
    Case("small_predicates", classify_text, {"value": STR}, "supported"),
    Case(
        "small_case_mapping",
        lambda value: value.title().swapcase().capitalize().casefold(),
        {"value": STR},
        "supported",
    ),
    Case(
        "small_strip",
        lambda value, characters: value.strip(characters).lstrip("ab").rstrip(characters),
        {"value": STR, "characters": STR.or_none()},
        "supported",
    ),
    Case(
        "small_tuple_affixes",
        lambda value: value.strip().startswith(("a", "b")) | value.endswith(("c", "d")),
        {"value": STR},
        "supported",
    ),
    Case(
        "small_boolean",
        lambda first, second, third: (first & second) | third,
        dict.fromkeys(("first", "second", "third"), BOOL),
        "supported",
    ),
    Case("small_list_membership", local_membership, {"value": INT.or_none()}, "supported"),
    Case(
        "small_extrema",
        lambda first, second, third: min(max(first, second, third), second, third),
        dict.fromkeys(("first", "second", "third"), FLOAT),
        "supported",
    ),
    Case(
        "small_padding",
        lambda value, width: value.zfill(width % 17).center(width % 31, "*"),
        {"value": STR, "width": INT},
        "supported",
    ),
]
