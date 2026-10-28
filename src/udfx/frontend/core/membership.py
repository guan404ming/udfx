from udfx.frontend.core.sequences import unit_stride
from udfx.graphs import deterministic
from udfx.ir import FALSE, Call, Lit, Node, lit
from udfx.ops.registry import apply
from udfx.semantics.casing import initial_lowercase_values
from udfx.types import STR, Kind, list_of


def comparison_membership(node: Call) -> Node | None:
    negative = node.op.name == "and"
    leaves: list[Node] = []

    def collect(part: Node) -> None:
        if isinstance(part, Call) and part.op.name == node.op.name:
            for argument in part.arguments:
                collect(argument)
        else:
            leaves.append(part)

    collect(node)
    source = None
    values = []
    for leaf in leaves:
        part = leaf
        negate = negative
        if isinstance(part, Call) and part.op.name == "not":
            part = part.arguments[0]
            negate = not negate
        if not isinstance(part, Call) or part.op.name != ("__ne__" if negate else "__eq__"):
            return None
        left, right = part.arguments
        if isinstance(left, Lit):
            left, right = right, left
        if (
            left.type.kind is not Kind.STR
            or not isinstance(right, Lit)
            or not isinstance(right.value, str)
            or (source is not None and source != left)
        ):
            return None
        source = left
        values.append(right.value)
    if source is None or not deterministic(source):
        return None
    result = apply("__contains__", Lit(tuple(dict.fromkeys(values)), list_of(STR)), source)
    return apply("not", result) if negative else result


def initial_lowercase_membership(node: Call) -> Node | None:
    container, value = node.arguments
    if (
        not isinstance(container, Lit)
        or not isinstance(container.value, tuple | list)
        or not all(isinstance(item, str) for item in container.value)
        or not isinstance(value, Call)
        or value.op.name != "__add__"
        or value.type != STR
    ):
        return None
    first, tail = value.arguments
    if not isinstance(first, Call) or first.op.name != ".lower":
        return None
    head = first.arguments[0]
    if (
        not isinstance(head, Call)
        or head.op.name != "__getitem__"
        or head.arguments[1] != lit(0)
        or not isinstance(tail, Call)
        or tail.op.name != "slice"
    ):
        return None
    source, start, stop, *steps = tail.arguments
    if (
        source != head.arguments[0]
        or start != lit(1)
        or not isinstance(stop, Lit)
        or stop.value is not None
        or not unit_stride(steps)
    ):
        return None
    values = initial_lowercase_values(tuple(container.value))
    return FALSE if not values else apply("__contains__", Lit(values, container.type), source)
