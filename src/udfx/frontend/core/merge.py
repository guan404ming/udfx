from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

from udfx.frontend.core.evaluation import Return
from udfx.frontend.core.logic import (
    choose,
    coalesce_branch,
    conjunction,
    conjuncts,
    negate,
    retype,
    unified,
)
from udfx.ir import TRUE, Call, If, Node
from udfx.types import Type

Case = tuple[tuple[Node, ...], Node | None]


def merge(returns: Sequence[Return], faults: Sequence[Node]) -> Node:
    result = unified([outcome.value for outcome in returns])
    cases: list[Case] = [(_path_predicates(outcome.path), outcome.value) for outcome in returns]
    cases += [(_path_predicates(path), None) for path in faults]
    return _build(cases, result)


def _path_predicates(path: Node) -> tuple[Node, ...]:
    return tuple(part for part in conjuncts(path) if part != TRUE)


def _atom(literal: Node) -> Node:
    if isinstance(literal, Call) and literal.op.name == "not":
        return literal.arguments[0]
    return literal


def _sign(atom: Node, literals: tuple[Node, ...]) -> bool | None:
    if all(part in literals for part in conjuncts(atom)):
        return True
    if negate(atom) in literals:
        return False
    return None


def _mentions(atom: Node, cases: list[Case]) -> int:
    return sum(_sign(atom, literals) is not None for literals, _ in cases)


def _pick(cases: list[Case]) -> Node | None:
    atoms = dict.fromkeys(_atom(literal) for literals, _ in cases for literal in literals)
    useful = [
        atom
        for atom in atoms
        if len({_sign(atom, literals) for literals, value in cases if value is not None}) > 1
    ]
    return max(useful, key=lambda atom: _mentions(atom, cases), default=None)


def _side(cases: list[Case], atom: Node, sign: bool) -> list[Case]:
    decided = conjuncts(atom) if sign else (negate(atom),)
    side: list[Case] = []
    for literals, value in cases:
        mention = _sign(atom, literals)
        if mention is None:
            side.append((literals, value))
        elif mention == sign:
            side.append((tuple(part for part in literals if part not in decided), value))
    return side


def _build(cases: list[Case], result: Type) -> Node:
    returns = [(literals, value) for literals, value in cases if value is not None]
    atom = None if len(returns) == 1 else _pick(cases)
    if atom is None:
        chain = [(reduce(conjunction, literals, TRUE), value) for literals, value in returns]
        return retype(choose(chain), result)
    positive = _build(_side(cases, atom, True), result)
    negative = _build(_side(cases, atom, False), result)
    return coalesce_branch(If(atom, positive, negative, result))
