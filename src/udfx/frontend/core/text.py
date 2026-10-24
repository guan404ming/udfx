from udfx.frontend.core.sequences import unit_stride
from udfx.ir import Call, Guarded, Lit, Node, lit
from udfx.ops.registry import apply
from udfx.semantics.casing import folded_case
from udfx.semantics.text import CASE_METHODS


def case_chain(node: Call) -> tuple[Node, list[str]]:
    source = node.arguments[0]
    methods = [node.op.name.removeprefix(".")]
    while isinstance(source, Call) and source.op.name in {f".{method}" for method in CASE_METHODS}:
        methods.append(source.op.name.removeprefix("."))
        source = source.arguments[0]
    return source, methods


def guarded_case_chain(node: Call) -> Node:
    source, methods = case_chain(node)
    if source == node.arguments[0] or isinstance(source, Lit):
        return node
    if folded_case(methods) == ():
        return apply(".casefold", source)
    return Guarded(apply(".lower", source), apply(".isascii", source))


def folded_case_chain(node: Call) -> Node:
    source, methods = case_chain(node)
    if source == node.arguments[0] or isinstance(source, Lit):
        return node
    replacements = folded_case(methods)
    if replacements is None:
        return guarded_case_chain(node)
    for original, replacement in replacements:
        source = apply(".replace", source, lit(original), lit(replacement))
    return apply(".casefold", source)


def simplify_text(node: Call) -> Node:
    if node.op.name == ".casefold":
        source, methods = case_chain(node)
        if source != node.arguments[0] and folded_case(methods) == ():
            return apply(".casefold", source)
    if node.op.name == ".lower":
        guarded = guarded_case_chain(node)
        if guarded != node:
            return guarded
    if node.op.name in {".lower", ".upper"}:
        argument = node.arguments[0]
        if (
            isinstance(argument, Call)
            and argument.op.name in {".strip", ".lstrip", ".rstrip"}
            and len(argument.arguments) == 1
        ):
            return apply(argument.op.name, apply(node.op.name, argument.arguments[0]))
    return last_split(node)


def last_split(node: Call) -> Node:
    if node.op.name != "slice":
        return node
    text, start, stop, *steps = node.arguments
    if not unit_stride(steps):
        return node
    if not isinstance(stop, Lit) or stop.value is not None:
        return node
    if not isinstance(start, Call) or start.op.name != "__add__":
        return node
    search, offset = start.arguments
    if not isinstance(search, Call) or search.op.name != ".rfind":
        return node
    source, pattern = search.arguments
    if (
        source == text
        and isinstance(pattern, Lit)
        and isinstance(pattern.value, str)
        and len(pattern.value) == 1
        and isinstance(offset, Lit)
        and offset.value == 1
    ):
        return apply("last_split", text, pattern)
    return node
