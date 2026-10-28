from udfx.ir import Call, If, Lit, Node
from udfx.semantics.assumptions import CONDITIONS
from udfx.semantics.bindings import source
from udfx.semantics.regex_replacement import replacement_has_newline

PRESERVING = frozenset(
    {
        ".lower",
        ".upper",
        ".casefold",
        ".capitalize",
        ".title",
        ".swapcase",
        ".strip",
        ".lstrip",
        ".rstrip",
        "slice",
        "__getitem__",
        "re.search",
        "re.match",
        "re.fullmatch",
    }
)


def argument_without_newline(node: Call) -> bool:
    seen: dict[int, bool] = {}

    def visit(node: Node) -> bool:
        node = source(node)
        if id(node) not in seen:
            seen[id(node)] = False
            seen[id(node)] = inspect(node)
        return seen[id(node)]

    def inspect(node: Node) -> bool:
        if isinstance(node, Lit):
            return isinstance(node.value, str) and "\n" not in node.value
        if isinstance(node, If):
            return visit(node.then) and visit(node.orelse)
        for predicate, truth in CONDITIONS.get():
            if (
                not truth
                and isinstance(predicate, Call)
                and predicate.op.name == "__contains__"
                and source(predicate.arguments[0]) == node
                and isinstance(predicate.arguments[1], Lit)
                and predicate.arguments[1].value == "\n"
            ):
                return True
        if not isinstance(node, Call):
            return False
        if node.op.name in PRESERVING:
            return visit(node.arguments[0])
        if node.op.name == "__add__":
            return all(visit(argument) for argument in node.arguments)
        if node.op.name == ".replace":
            return visit(node.arguments[0]) and visit(node.arguments[2])
        if node.op.name == "re.sub":
            pattern, flags, replacement = node.arguments[1:4]
            return (
                visit(node.arguments[0])
                and isinstance(pattern, Lit)
                and isinstance(pattern.value, str)
                and isinstance(flags, Lit)
                and isinstance(flags.value, int)
                and isinstance(replacement, Lit)
                and isinstance(replacement.value, str)
                and not replacement_has_newline(pattern.value, flags.value, replacement.value)
            )
        return False

    return visit(node.arguments[0])
