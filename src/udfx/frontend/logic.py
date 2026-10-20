from __future__ import annotations

from collections.abc import Sequence

from udfx.errors import Unsupported
from udfx.ir import FALSE, TRUE, Call, If, Lit, Node
from udfx.ops import apply
from udfx.types import Kind, Type, fits, unify

EXCLUSIVE = frozenset({"bool", "is_none"})


def fold(node: Node) -> Node:
    if not isinstance(node, Call):
        return node
    folded = Call(node.op, tuple(fold(argument) for argument in node.args))
    values = [argument.value for argument in folded.args if isinstance(argument, Lit)]
    if len(values) < len(folded.args):
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
        return conjuncts(node.args[0]) + conjuncts(node.args[1])
    return (node,)


def refuted(part: Node, known: Sequence[Node]) -> bool:
    if negate(part) in known:
        return True
    if not isinstance(part, Call) or part.op.name not in EXCLUSIVE:
        return False
    return any(
        isinstance(fact, Call)
        and fact.op.name in EXCLUSIVE - {part.op.name}
        and fact.args == part.args
        for fact in known
    )


def conjunction(a: Node, b: Node) -> Node:
    if a == TRUE:
        return b
    if b == TRUE:
        return a
    if FALSE in (a, b):
        return FALSE
    known = list(conjuncts(a))
    result = a
    for part in conjuncts(b):
        if refuted(negate(part), known):
            continue
        if refuted(part, known):
            return FALSE
        known.append(part)
        result = apply("and", result, part)
    return result


def disjunction(a: Node, b: Node) -> Node:
    if a in (FALSE, b):
        return b
    if b == FALSE:
        return a
    if TRUE in (a, b):
        return TRUE
    return apply("or", a, b)


def negate(a: Node) -> Node:
    if a == TRUE:
        return FALSE
    if a == FALSE:
        return TRUE
    if isinstance(a, Call) and a.op.name == "not":
        return a.args[0]
    return apply("not", a)


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
