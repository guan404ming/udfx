from collections.abc import Sequence
from graphlib import CycleError, TopologicalSorter
from typing import Any

from udfx.errors import Unsupported
from udfx.semantics.literals import TextPrimitives
from udfx.semantics.text import char_class


def translation_table(table: object) -> tuple[str, tuple[str, ...]]:
    if type(table) is not dict:
        raise Unsupported("translate requires a constant dict")
    characters: list[str] = []
    replacements: list[str] = []
    for key, value in table.items():
        if type(key) not in (int, float, bool, str, type(None)):
            raise Unsupported("translate requires scalar constant keys")
        if not isinstance(key, int | float) or not 0 <= key <= 0x10FFFF or int(key) != key:
            continue
        if value is None:
            replacement = ""
        elif isinstance(value, str):
            replacement = value
        elif isinstance(value, int) and 0 <= value <= 0x10FFFF:
            replacement = chr(value)
        else:
            raise Unsupported("translate requires string, ordinal, or None replacements")
        characters.append(chr(int(key)))
        replacements.append(replacement)
    return "".join(characters), tuple(replacements)


def translate_text(text: str, characters: str, replacements: Sequence[str]) -> str:
    return text.translate(dict(zip(map(ord, characters), replacements, strict=False)))


def lower_translation(
    primitives: TextPrimitives, text: Any, characters: Any, replacements: Any
) -> tuple[Any, Any]:
    if not isinstance(characters, str) or not isinstance(replacements, tuple | list):
        return text, primitives.char_length(text) < 0
    mapping = dict(zip(characters, replacements, strict=False))
    if not all(isinstance(value, str) for value in mapping.values()):
        return text, primitives.char_length(text) < 0
    if not mapping:
        return text, None
    if all(len(value) <= 1 for value in mapping.values()):
        ordered = sorted(mapping, key=lambda character: not mapping[character])
        return primitives.translate(
            text, "".join(ordered), "".join(mapping[character] for character in ordered)
        ), None
    dependencies = {
        character: (set(replacement) & mapping.keys()) - {character}
        for character, replacement in mapping.items()
    }
    try:
        ordered = tuple(TopologicalSorter(dependencies).static_order())
    except CycleError:
        ordered = ()
    if ordered:
        for character in ordered:
            text = primitives.replace(text, character, mapping[character])
        return text, None
    used = set(mapping) | set("".join(mapping.values()))
    markers = [chr(point) for point in range(0xE000, 0xF900) if chr(point) not in used][
        : len(mapping)
    ]
    if len(markers) < len(mapping):
        return text, primitives.char_length(text) < 0
    result = text
    for character, marker in zip(mapping, markers, strict=True):
        result = primitives.replace(result, character, marker)
    for replacement, marker in zip(mapping.values(), markers, strict=True):
        result = primitives.replace(result, marker, replacement)
    return result, ~primitives.matches(text, char_class(markers))
