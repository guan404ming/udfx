from collections.abc import Sequence

from udfx.frontend.core.logic import fold
from udfx.graphs import canonicalize, canonicalizer, map_children
from udfx.ir import FALSE, TRUE, Call, If, Lit, Node, lit
from udfx.ops import Op, apply
from udfx.types import BOOL, STR


def concatenation(node: Node) -> Call | None:
    return (
        node
        if isinstance(node, Call)
        and node.op.name == "__add__"
        and node.op.deterministic
        and node.type == STR
        and all(argument.type == STR for argument in node.arguments)
        else None
    )


def common_edges(
    first: Sequence[Node], second: Sequence[Node], op: Op, *, reverse: bool = False
) -> tuple[list[Node], list[Node], list[Node]]:
    left = list(first if reverse else reversed(first))
    right = list(second if reverse else reversed(second))
    expanded: tuple[set[int], set[int]] = (set(), set())
    common: list[Node] = []
    while left and right:
        if left[-1] is right[-1]:
            common.append(left.pop())
            right.pop()
            continue
        changed = False
        for pending, seen in zip((left, right), expanded, strict=True):
            node = pending[-1]
            addition = concatenation(node)
            if addition is not None and addition.op is op and id(node) not in seen:
                seen.add(id(node))
                pending.pop()
                pending.extend(addition.arguments if reverse else reversed(addition.arguments))
                changed = True
        if not changed:
            break
    return (
        list(reversed(common)) if reverse else common,
        left if reverse else list(reversed(left)),
        right if reverse else list(reversed(right)),
    )


def join_strings(parts: Sequence[Node], op: Op) -> Node:
    if not parts:
        return lit("")
    if len(parts) == 1:
        return parts[0]
    middle = len(parts) // 2
    return Call(op, (join_strings(parts[:middle], op), join_strings(parts[middle:], op)))


def shared_branch(node: If) -> If:
    if node.condition.type != BOOL:
        return node
    left, right = node.then, node.orelse
    if isinstance(left, If) and left.condition.type == BOOL and not isinstance(right, Lit):
        if left.orelse is right:
            predicate = If(node.condition, left.condition, FALSE, BOOL)
            return If(predicate, left.then, right, node.type)
        if left.then is right:
            predicate = If(node.condition, left.condition, TRUE, BOOL)
            return If(predicate, right, left.orelse, node.type)
    if isinstance(right, If) and right.condition.type == BOOL and not isinstance(left, Lit):
        if right.then is left:
            predicate = If(node.condition, TRUE, right.condition, BOOL)
            return If(predicate, left, right.orelse, node.type)
        if right.orelse is left:
            predicate = If(node.condition, FALSE, right.condition, BOOL)
            return If(predicate, right.then, left, node.type)
    return node


def factor(node: Node) -> Node:
    if not isinstance(node, If):
        return node
    node = shared_branch(node)
    left, right = node.then, node.orelse
    if left is right:
        return left
    if (
        node.condition.type == BOOL
        and isinstance(left, Lit)
        and isinstance(right, Lit)
        and left.type == right.type == BOOL
    ):
        return node.condition if left.value is True else apply("not", node.condition)
    if node.type != STR or left.type != STR or right.type != STR:
        return node
    addition = concatenation(left) or concatenation(right)
    if addition is None:
        return node
    prefix, first, second = common_edges((left,), (right,), addition.op)
    suffix, first, second = common_edges(first, second, addition.op, reverse=True)
    if not prefix and not suffix:
        return node
    selected = If(
        node.condition,
        join_strings(first, addition.op),
        join_strings(second, addition.op),
        STR,
    )
    return join_strings((*prefix, selected, *suffix), addition.op)


def factor_branches(roots: Sequence[Node]) -> tuple[Node, ...]:
    intern = canonicalizer()
    seen: dict[int, Node] = {}

    def visit(node: Node) -> Node:
        if id(node) not in seen:
            seen[id(node)] = intern(factor(map_children(node, visit)))
        return seen[id(node)]

    roots = tuple(intern(node) for node in roots)

    return canonicalize(tuple(fold(visit(node)) for node in roots))
