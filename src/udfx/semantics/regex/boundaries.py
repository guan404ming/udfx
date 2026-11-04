from collections.abc import Callable
from typing import Any


def ending_classes(token: Any) -> list[list[Any]] | None:
    name, value = token
    if str(name) in {"MAX_REPEAT", "MIN_REPEAT"} and value[0] > 0 and value[2]:
        return ending_classes(value[2][-1])
    if str(name) == "SUBPATTERN" and value[3].getwidth()[0] > 0:
        return ending_classes(value[3][-1])
    members = value if str(name) == "IN" else [token]
    parts: list[list[Any]] = [[], []]
    for name, value in members:
        if str(name) == "LITERAL":
            word = chr(value).isalnum() or chr(value) == "_"
        elif str(name) == "CATEGORY" and str(value) in {
            "CATEGORY_DIGIT",
            "CATEGORY_WORD",
            "CATEGORY_SPACE",
            "CATEGORY_NOT_WORD",
        }:
            word = str(value) in {"CATEGORY_DIGIT", "CATEGORY_WORD"}
        else:
            return None
        parts[int(word)].append((name, value))
    return parts


def single_unit(members: list[Any]) -> bool:
    return all(
        (str(name) == "LITERAL" and value < 0x10000)
        or (str(name) == "CATEGORY" and str(value) == "CATEGORY_SPACE")
        for name, value in members
    )


def anchor_after(
    token: Any, boundary: bool, word: str, character_set: Callable[[Any], str]
) -> str | None:
    previous = ending_classes(token)
    if previous is None or (all(previous) and not any(map(single_unit, previous))):
        return None
    choices = []
    for matched, members in enumerate(previous):
        if not members:
            continue
        ahead = "!" if bool(matched) == boundary else "="
        behind = ""
        if all(previous):
            native = single_unit(members)
            selected = members if native else previous[1 - matched]
            behind = "(?<" + ("=" if native else "!") + character_set(selected) + ")"
        choices.append(behind + "(?" + ahead + word + ")")
    return "(?:" + "|".join(choices) + ")"
