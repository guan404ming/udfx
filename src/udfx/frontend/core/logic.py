from __future__ import annotations

from collections.abc import Sequence

from udfx.errors import Unsupported
from udfx.frontend.core.sequences import simplify_sequence
from udfx.frontend.core.text import simplify_text
from udfx.ir import FALSE, TRUE, Call, Comprehension, Element, Guarded, If, ListValue, Lit, Node
from udfx.ops import apply
from udfx.types import Kind, Type, fits, unify

EXCLUSIVE = frozenset({"bool", "is_none"})


def fold(node: Node) -> Node:
    if not isinstance(node, Call):
        return node
    folded = simplify_sequence(Call(node.op, tuple(fold(argument) for argument in node.arguments)))
    if isinstance(folded, Call):
        folded = simplify_text(folded)
    if not isinstance(folded, Call):
        return folded
    values = [argument.value for argument in folded.arguments if isinstance(argument, Lit)]
    if len(values) < len(folded.arguments):
        return folded
    if folded.type.kind is Kind.LIST and folded.op.name in {"range", "__mul__"}:
        return folded
    try:
        value = folded.op.reference(*values)
    except Exception:
        return folded
    if not fits(value, folded.type):
        return folded
    return Lit(tuple(value) if isinstance(value, list) else value, folded.type)


def conjuncts(node: Node) -> tuple[Node, ...]:
    if isinstance(node, Call) and node.op.name == "and":
        return conjuncts(node.arguments[0]) + conjuncts(node.arguments[1])
    return (node,)


def refuted(part: Node, known: Sequence[Node]) -> bool:
    if negate(part) in known:
        return True
    if not isinstance(part, Call) or part.op.name not in EXCLUSIVE:
        return False
    return any(
        isinstance(fact, Call)
        and fact.op.name in EXCLUSIVE - {part.op.name}
        and fact.arguments == part.arguments
        for fact in known
    )


def conjunction(left: Node, right: Node) -> Node:
    if left == TRUE:
        return right
    if right == TRUE:
        return left
    if FALSE in (left, right):
        return FALSE
    known = list(conjuncts(left))
    result = left
    for part in conjuncts(right):
        if refuted(negate(part), known):
            continue
        if refuted(part, known):
            return FALSE
        known.append(part)
        result = apply("and", result, part)
    return result


def disjunction(left: Node, right: Node) -> Node:
    if left in (FALSE, right):
        return right
    if right == FALSE:
        return left
    if TRUE in (left, right):
        return TRUE
    return apply("or", left, right)


def negate(value: Node) -> Node:
    if value == TRUE:
        return FALSE
    if value == FALSE:
        return TRUE
    if isinstance(value, Call) and value.op.name == "not":
        return value.arguments[0]
    return apply("not", value)


def unified(values: Sequence[Node]) -> Type:
    result = values[0].type
    for value in values[1:]:
        joined = unify(result, value.type)
        if joined is None:
            raise Unsupported(
                f"return types differ: returns {result} on one path and {value.type} "
                "on another; return one type"
            )
        result = joined
    return result


def choose(cases: Sequence[tuple[Node, Node]]) -> Node:
    result = unified([value for _, value in cases])
    *rest, (_, value) = cases
    node = retype(value, result)
    for condition, then in reversed(rest):
        node = If(condition, retype(then, result), node, result)
    return node


def retype(node: Node, type: Type) -> Node:
    if isinstance(node, Lit) and node.type.kind is Kind.NONE:
        return Lit(None, type.or_none())
    return node


def depends_on(node: Node, element: Element) -> bool:
    match node:
        case Element():
            return node == element
        case Call():
            children = node.arguments
        case Guarded():
            children = (node.value, node.exact)
        case If():
            children = (node.condition, node.then, node.orelse)
        case ListValue():
            children = node.items
        case Comprehension():
            children = (node.source, node.value, node.condition, node.fault)
        case _:
            return False
    return any(depends_on(child, element) for child in children)
