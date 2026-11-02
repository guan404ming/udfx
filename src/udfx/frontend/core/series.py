from __future__ import annotations

from udfx.errors import Unsupported
from udfx.frontend.core.logic import disjunction, negate
from udfx.frontend.core.tracing import SERIES
from udfx.frontend.core.values import Series
from udfx.graphs import map_children
from udfx.ir import Call, Col, Comprehension, Element, Node, children
from udfx.types import list_of

AGGREGATE = "udfx aggregate "


def rows() -> dict[str, Node]:
    known = SERIES.get()
    if known is None:
        raise Unsupported("pandas.Series requires a series UDF")
    return known


def marker(series: Series) -> Col:
    known = rows()
    name = f"udfx series {len(known)}"
    known[name] = series.node
    return Col(name, list_of(series.node.type))


def batch_length(node: Node, known: dict[str, Node]) -> bool:
    if isinstance(node, Call) and node.op.name == "min":
        return all(batch_length(argument, known) for argument in node.arguments)
    return (
        isinstance(node, Call)
        and node.op.name == "len"
        and isinstance(node.arguments[0], Col)
        and node.arguments[0].name in known
    )


def per_row(node: Node, element: Element, row: Node | None, known: dict[str, Node]) -> Node:
    seen: dict[int, Node] = {}

    def visit(part: Node) -> Node:
        if id(part) not in seen:
            if part == element:
                if row is None:
                    raise Unsupported("series values depend on their batch position")
                seen[id(part)] = row
            elif (
                isinstance(part, Call)
                and part.op.name == "__getitem__"
                and part.arguments[1] == element
                and isinstance(part.arguments[0], Col)
                and part.arguments[0].name in known
            ):
                seen[id(part)] = known[part.arguments[0].name]
            elif isinstance(part, Col) and part.name in known:
                raise Unsupported("series values depend on the whole batch")
            else:
                seen[id(part)] = map_children(part, visit)
        return seen[id(part)]

    return visit(node)


def elementwise(node: Node) -> tuple[Node, Node]:
    known = rows()
    if not isinstance(node, Comprehension):
        raise Unsupported("pandas.Series requires one value per series row")
    source = node.source
    if isinstance(source, Col) and source.name in known:
        row = known[source.name]
    elif (
        isinstance(source, Call)
        and source.op.name == "range"
        and len(source.arguments) == 1
        and batch_length(source.arguments[0], known)
    ):
        row = None
    else:
        raise Unsupported("pandas.Series requires one value per series row")
    fault = disjunction(node.fault, negate(node.condition))
    return per_row(node.value, node.element, row, known), per_row(fault, node.element, row, known)


def require_reduced(*nodes: Node) -> None:
    pending = list(nodes)
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Col) and not node.name.startswith(AGGREGATE):
            raise Unsupported("aggregate UDFs must reduce every series")
        pending.extend(children(node))
