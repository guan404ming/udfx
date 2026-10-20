from __future__ import annotations

from udfx.ir import FALSE, TRUE, Node
from udfx.ops import apply


def conj(a: Node, b: Node) -> Node:
    if a == TRUE:
        return b
    if b == TRUE:
        return a
    if FALSE in (a, b):
        return FALSE
    return apply("and", a, b)


def disj(a: Node, b: Node) -> Node:
    if a == FALSE:
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
    return apply("not", a)
