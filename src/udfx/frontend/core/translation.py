from typing import cast

from udfx.ir import Call, Lit, Node, lit
from udfx.ops import apply
from udfx.semantics.text.translation import compose_replacement
from udfx.types import STR, list_of


def literal_translation(node: Node) -> tuple[Node, str, tuple[str, ...]] | None:
    if not isinstance(node, Call) or node.op.name not in {".replace", ".translate"}:
        return None
    source, characters, replacements = node.arguments
    if not isinstance(characters, Lit) or not isinstance(characters.value, str):
        return None
    if not isinstance(replacements, Lit):
        return None
    values = replacements.value
    if node.op.name == ".replace":
        if len(characters.value) != 1:
            return None
        values = (values,)
    if (
        not isinstance(values, tuple)
        or len(values) != len(characters.value)
        or any(not isinstance(value, str) or len(value) > 1 for value in values)
    ):
        return None
    return source, characters.value, cast(tuple[str, ...], values)


def combined_replacements(node: Call) -> Node:
    if node.op.name != ".replace":
        return node
    source: Node = node
    changes = []
    while isinstance(source, Call) and source.op.name == ".replace":
        translated = literal_translation(source)
        if translated is None:
            break
        source, old, replacements = translated
        changes.append((old, replacements[0]))
    translated = literal_translation(source)
    if translated is not None:
        source, characters, values = translated
    elif len(changes) > 1:
        characters, replacement = changes.pop()
        values = (replacement,)
    else:
        return node
    for old, new in reversed(changes):
        characters, values = compose_replacement(characters, values, old, new)
    return apply(".translate", source, lit(characters), Lit(values, list_of(STR)))
