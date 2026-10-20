from __future__ import annotations

from collections.abc import Sequence

from udfx.errors import Unsupported
from udfx.ir import FALSE, TRUE, Call, If, Lit, Node
from udfx.ops import apply
from udfx.types import Kind, Type, unify

EXCLUSIVE = frozenset({"bool", "is_none"})


def conjuncts(node: Node) -> tuple[Node, ...]:
    if isinstance(node, Call) and node.op.name == "and":
        return conjuncts(node.args[0]) + conjuncts(node.args[1])
    return (node,)


def refuted(part: Node, known: Sequence[Node]) -> bool:
    if neg(part) in known:
        return True
    if not isinstance(part, Call) or part.op.name not in EXCLUSIVE:
        return False
    return any(
        isinstance(fact, Call)
        and fact.op.name in EXCLUSIVE - {part.op.name}
        and fact.args == part.args
        for fact in known
    )


def conj(a: Node, b: Node) -> Node:
    if a == TRUE:
        return b
    if b == TRUE:
        return a
    if FALSE in (a, b):
        return FALSE
    known = list(conjuncts(a))
    result = a
    for part in conjuncts(b):
        if refuted(neg(part), known):
            continue
        if refuted(part, known):
            return FALSE
        known.append(part)
        result = apply("and", result, part)
    return result


def disj(a: Node, b: Node) -> Node:
    if a in (FALSE, b):
        return b
    if b == FALSE:
        return a
    if TRUE in (a, b):
        return TRUE
    return apply("or", a, b)


def neg(a: Node) -> Node:
    if a == TRUE:
        return FALSE
    if a == FALSE:
        return TRUE
    if isinstance(a, Call) and a.op.name == "not":
        return a.args[0]
    return apply("not", a)


def choose(cases: Sequence[tuple[Node, Node]]) -> Node:
    result = cases[0][1].type
    for _, value in cases[1:]:
        unified = unify(result, value.type)
        if unified is None:
            raise Unsupported(f"types {result} and {value.type} differ")
        result = unified
    *rest, (_, value) = cases
    node = _retype(value, result)
    for cond, then in reversed(rest):
        node = If(cond, _retype(then, result), node, result)
    return node


def _retype(node: Node, type: Type) -> Node:
    if isinstance(node, Lit) and node.type.kind is Kind.NONE:
        return Lit(None, type.or_none())
    return node
