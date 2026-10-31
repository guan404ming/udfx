from __future__ import annotations

from collections.abc import Sequence

from udfx.errors import Unsupported
from udfx.frontend.core.lists import optional_item, simplify_list
from udfx.frontend.core.numbers import simplify_number
from udfx.frontend.core.sequences import simplify_sequence
from udfx.frontend.core.sets import simplify_set
from udfx.frontend.core.text import simplify_text
from udfx.frontend.core.tracing import FOLDED
from udfx.graphs import deterministic
from udfx.ir import FALSE, TRUE, Call, Comprehension, Element, Guarded, If, ListValue, Lit, Node
from udfx.ops import apply
from udfx.types import JSON, Kind, Type, fits, unify

EXCLUSIVE = frozenset({"bool", "is_none"})


def fold(node: Node) -> Node:
    seen = FOLDED.get()
    return fold_shared(node, {} if seen is None else seen)


def fold_shared(node: Node, seen: dict[int, tuple[Node, Node]]) -> Node:
    found = seen.get(id(node))
    if found is None:
        found = node, fold_node(node, seen)
        seen[id(node)] = found
    return found[1]


def fold_node(node: Node, seen: dict[int, tuple[Node, Node]]) -> Node:
    if isinstance(node, If):
        condition = fold_shared(node.condition, seen)
        if isinstance(condition, Lit):
            return retype(
                fold_shared(node.then if condition.value else node.orelse, seen), node.type
            )
        return simplify_branch(
            If(condition, fold_shared(node.then, seen), fold_shared(node.orelse, seen), node.type)
        )
    if not isinstance(node, Call):
        return node
    arguments = tuple(fold_shared(argument, seen) for argument in node.arguments)
    folded = simplify_sequence(Call(node.op, arguments))
    if isinstance(folded, Call):
        folded = simplify_list(folded)
    if isinstance(folded, Call):
        folded = simplify_text(folded)
    if isinstance(folded, Call):
        folded = simplify_set(folded)
    if isinstance(folded, Call):
        folded = simplify_number(folded)
    if not isinstance(folded, Call) or not folded.op.deterministic:
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


def simplify_branch(node: If) -> Node:
    selected = optional_item(node)
    if selected is not None:
        return selected
    predicate = node.condition
    positive = True
    if isinstance(predicate, Call) and predicate.op.name == "not":
        predicate = predicate.arguments[0]
        positive = False
    if not isinstance(predicate, Call):
        return node
    if predicate.op.name == "is_none":
        source = predicate.arguments[0]
        positive = not positive
    elif predicate.op.name in {"__eq__", "__ne__"} and any(
        isinstance(part, Lit) and part.value is None for part in predicate.arguments
    ):
        left, right = predicate.arguments
        source = right if isinstance(left, Lit) and left.value is None else left
        positive = positive == (predicate.op.name == "__ne__")
    else:
        return node
    value, default = (node.then, node.orelse) if positive else (node.orelse, node.then)
    if (
        value != source
        or value.type.kind is not Kind.STR
        or not isinstance(default, Lit)
        or not isinstance(default.value, str)
        or not deterministic(source)
    ):
        return node
    return apply("coalesce", source, default)


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
        node = fold(If(condition, retype(then, result), node, result))
    return node


def retype(node: Node, type: Type) -> Node:
    if type.kind is Kind.JSON and node.type.kind is not Kind.JSON:
        return Lit("null", JSON) if node.type.kind is Kind.NONE else apply("json.encode", node)
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
