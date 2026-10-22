from udfx.ir import Call, Lit
from udfx.ops.registry import apply


def simplify_text(node: Call) -> Call:
    if node.op.name in {".lower", ".upper"}:
        argument = node.arguments[0]
        if (
            isinstance(argument, Call)
            and argument.op.name in {".strip", ".lstrip", ".rstrip"}
            and len(argument.arguments) == 1
        ):
            return apply(argument.op.name, apply(node.op.name, argument.arguments[0]))
    if node.op.name != "slice":
        return node
    text, start, stop, *steps = node.arguments
    if steps and not (isinstance(steps[0], Lit) and steps[0].value in (None, 1)):
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
