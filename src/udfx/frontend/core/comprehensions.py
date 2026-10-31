from collections.abc import Callable
from dataclasses import replace

from udfx.frontend.core.reductions import substitute
from udfx.graphs import deterministic
from udfx.ir import FALSE, TRUE, Call, Comprehension, Element, Guarded, Lit, Node, children
from udfx.ops.registry import apply


def uses(node: Node, element: Element, linear: Callable[[Call], bool]) -> int:
    if node == element:
        return 1
    if isinstance(node, Comprehension):
        source = uses(node.source, element, linear)
        if node.element == element:
            return source
        body = sum(uses(part, element, linear) for part in (node.value, node.condition, node.fault))
        return min(2, source + 2 * body)
    count = 0
    for part in children(node):
        count += uses(part, element, linear)
        if count >= 2:
            return 2
    return 2 if count and isinstance(node, Call) and not linear(node) else count


def total(node: Comprehension) -> bool:
    return node.fault == FALSE and deterministic(node.value)


def compose(
    node: Comprehension,
    linear: Callable[[Call], bool],
    rewrite: Callable[[Call], Node | None],
) -> Comprehension:
    while isinstance(source := node.source, Comprehension):
        if (
            node.condition != TRUE
            or source.condition != TRUE
            or not total(node)
            or not total(source)
        ):
            break
        value = substitute(node.value, node.element, source.value)
        if source.value != source.element and uses(node.value, node.element, linear) != 1:
            rewritten = rewrite(value) if isinstance(value, Call) else None
            if rewritten is None or rewritten == value:
                break
            value = rewritten
        node = replace(
            node,
            source=source.source,
            element=source.element,
            value=value,
            requires_native=node.requires_native or source.requires_native,
        )
    return node


def cardinality(node: Call) -> Node | None:
    source = node.arguments[0]
    if isinstance(source, Call):
        name = "re.count" if node.op.name == "len" else "re.contains"
        if source.op.name == "re.findall":
            return apply(name, *source.arguments)
        if source.op.name == "json.tokens":
            value = source.arguments[0]
            size = apply("json.length", apply("json.value", value))
            result = size if node.op.name == "len" else apply("bool", size)
            return Guarded(result, apply("json.array_native", value))
    if isinstance(source, Comprehension) and source.condition == TRUE and total(source):
        return apply(node.op.name, source.source)
    return None


def lowercase_words(node: Call) -> Node | None:
    words = node.arguments[0]
    while isinstance(words, Call) and words.op.name == "list":
        words = words.arguments[0]
    if not isinstance(words, Comprehension) or words.condition != TRUE or words.fault != FALSE:
        return None
    source, value = words.source, words.value
    if (
        not isinstance(source, Call)
        or source.op.name != ".split"
        or len(source.arguments) != 1
        or not isinstance(value, Call)
        or value.op.name != ".lower"
        or value.arguments != (words.element,)
    ):
        return None
    lowered = apply(".split", apply(".lower", source.arguments[0]))
    return replace(node, arguments=(lowered, *node.arguments[1:]))


def permutation_source(value: Node) -> Node:
    if isinstance(value, Call):
        if value.op.name in {"sorted", "list"}:
            return permutation_source(value.arguments[0])
        if value.op.name == ".join":
            separator, items = value.arguments
            items = permutation_source(items)
            if (
                isinstance(items, Call)
                and items.op.name in {".split", ".rsplit"}
                and len(items.arguments) >= 2
                and items.arguments[1] == separator
                and isinstance(separator, Lit)
                and isinstance(separator.value, str)
                and separator.value
            ):
                return permutation_source(items.arguments[0])
    if (
        isinstance(value, Comprehension)
        and value.condition == TRUE
        and value.fault == FALSE
        and permutation_source(value.value) == value.element
    ):
        return permutation_source(value.source)
    return value
