from udfx.ir import Call, Lit, Node, lit
from udfx.semantics.assumptions import CONDITIONS
from udfx.semantics.bindings import source as bound_source


def source(value: Node) -> Node:
    value = bound_source(value)
    while isinstance(value, Call) and value.op.name in {"json.value", "json.text"}:
        value = bound_source(value.arguments[0])
    return value


def container(node: Call) -> str | None:
    value = source(node.arguments[0])
    for predicate, truth in CONDITIONS.get():
        if (
            truth
            and isinstance(predicate, Call)
            and predicate.op.name == "json.is_type"
            and source(predicate.arguments[0]) == value
            and isinstance(name := predicate.arguments[1], Lit)
            and isinstance(name.value, str)
        ):
            return name.value
        if (
            isinstance(predicate, Call)
            and predicate.op.name in {"__eq__", "__ne__"}
            and truth == (predicate.op.name == "__eq__")
        ):
            for selected, prefix in (predicate.arguments, predicate.arguments[::-1]):
                if (
                    isinstance(selected, Call)
                    and selected.op.name == "__getitem__"
                    and source(selected.arguments[0]) == value
                    and selected.arguments[1] == lit(0)
                    and isinstance(prefix, Lit)
                    and isinstance(prefix.value, str)
                ):
                    return {"[": "list", "{": "dict", '"': "str"}.get(prefix.value)
    return None


def indexed_path(node: Call) -> tuple[Node | None, tuple[int, ...]]:
    value = node.arguments[0]
    positions = []
    while isinstance(value, Call) and value.op.name == "json.item":
        position = value.arguments[1]
        if not isinstance(position, Lit) or type(position.value) is not int or position.value < 0:
            break
        positions.append(position.value)
        value = value.arguments[0]
    return (value if positions else None), tuple(reversed(positions))
