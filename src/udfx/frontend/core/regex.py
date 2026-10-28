from udfx.ir import FALSE, Call, Comprehension, Lit, Node, lit
from udfx.ops.registry import apply
from udfx.semantics.regex_patterns import first_run


def first_match(node: Call) -> Node | None:
    if node.op.name != "bool" and not (
        node.op.name == "__getitem__" and node.arguments[1] == lit(0)
    ):
        return None
    items = node.arguments[0]
    if not isinstance(items, Comprehension) or items.value != items.element or items.fault != FALSE:
        return None
    source, predicate = items.source, items.condition
    if not isinstance(source, Call) or source.op.name != "re.findall":
        return None
    if (
        not isinstance(predicate, Call)
        or predicate.op.name != "__ne__"
        or predicate.arguments[1] != lit(None)
    ):
        return None
    matched = predicate.arguments[0]
    if (
        not isinstance(matched, Call)
        or matched.op.name != "re.fullmatch"
        or matched.arguments[0] != items.element
    ):
        return None
    text, run, flags = source.arguments
    member, options = matched.arguments[1:]
    if (
        not isinstance(run, Lit)
        or not isinstance(run.value, str)
        or not isinstance(member, Lit)
        or not isinstance(member.value, str)
        or not isinstance(flags, Lit)
        or not isinstance(flags.value, int)
        or flags != options
    ):
        return None
    pattern = first_run(run.value, member.value, flags.value)
    if pattern is None:
        return None
    found = apply("re.first", text, lit(pattern), flags)
    return apply("__ne__", found, lit(None)) if node.op.name == "bool" else found
