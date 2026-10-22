from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from udfx.frontend.core.logic import conjunction, fold, negate
from udfx.frontend.core.values import Item
from udfx.ir import FALSE, TRUE, Node
from udfx.ops import Fault


@dataclass(frozen=True)
class Evaluation:
    items: tuple[Item, ...]
    faults: Sequence[Fault] = ()


@dataclass(frozen=True)
class Return:
    path: Node
    value: Node


@dataclass(frozen=True)
class Raise:
    path: Node
    kind: type[Exception]


Outcome = Return | Raise


def single(item: Item, faults: Sequence[Fault] = ()) -> Evaluation:
    return Evaluation((item,), faults)


def always(kind: type[Exception]) -> Evaluation:
    return Evaluation((), (Fault(kind, TRUE),))


def guard(path: Node, faults: Sequence[Fault]) -> tuple[Node | None, list[Raise]]:
    failures: list[Raise] = []
    for fault in faults:
        when = fold(fault.when)
        failing = conjunction(path, when)
        if failing != FALSE:
            failures.append(Raise(failing, fault.kind))
        path = conjunction(path, negate(when))
    return (path if path != FALSE else None), failures
