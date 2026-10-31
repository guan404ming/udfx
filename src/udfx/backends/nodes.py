from __future__ import annotations

from typing import TYPE_CHECKING

from udfx.backends.graph_lists import lower_graph_comprehension
from udfx.backends.guarantees import checked, required
from udfx.backends.lowering import lower_call, lower_comprehension, lower_if
from udfx.backends.lowering_context import graph_enabled
from udfx.backends.rules import Lowered, Scope, fold_predicate
from udfx.errors import Unsupported
from udfx.frontend.core.comprehensions import compose
from udfx.ir import TRUE, Call, Col, Comprehension, Element, Guarded, If, ListValue, Lit, Node
from udfx.types import Kind, fits

if TYPE_CHECKING:
    from udfx.backends.base import Backend


def lower_node[E](backend: Backend[E], node: Node, elements: Scope[E]) -> Lowered[E]:
    if elements and node in elements:
        value = elements[node]
        return value if isinstance(value, Lowered) else Lowered(value)
    match node:
        case Element():
            raise Unsupported(f"unbound list element {node.name}")
        case Guarded():
            value = backend.lower(node.value, elements)
            predicate = fold_predicate(node.exact, backend.rules.rewrite_call)
            if predicate == TRUE:
                return value
            exact = backend.lower(predicate, elements)
            return value.map(
                lambda value: Lowered(
                    value.expr,
                    backend.conjunction(value.exact, backend.exact(exact), exact.expr),
                    value.overflow,
                    value.recover_overflow,
                    guarantees=value.guarantees + checked(exact),
                )
            )
        case ListValue():
            items = [backend.lower(item, elements) for item in node.items]
            return Lowered(
                backend.list_value([item.expr for item in items], node.type),
                backend.conjunction(*(backend.exact(item) for item in items)),
                guarantees=required(items),
            )
        case Comprehension():
            lower = lower_graph_comprehension if graph_enabled() else lower_comprehension
            return lower(
                backend, compose(node, backend.rules.linear, backend.rules.rewrite_call), elements
            )
        case Col():
            return Lowered(backend.column(node))
        case Lit() if (
            node.type.kind is Kind.LIST
            and node.type.item is not None
            and isinstance(node.value, tuple | list)
        ):
            return backend.lower(
                ListValue(tuple(Lit(item, node.type.item) for item in node.value), node.type),
                elements,
            )
        case Lit() if node.type.kind is Kind.INT and not fits(node.value, node.type):
            return Lowered(backend.literal(Lit(0, node.type)), overflow=backend.literal(TRUE))
        case Lit():
            return Lowered(backend.literal(node))
        case If():
            return lower_if(backend, node, elements)
        case Call():
            return lower_call(backend, node, elements)
