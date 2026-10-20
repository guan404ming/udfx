from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from udfx.types import BOOL, Type, type_of

if TYPE_CHECKING:
    from udfx.ops.registry import Op


@dataclass(frozen=True)
class Col:
    name: str
    type: Type


@dataclass(frozen=True)
class Lit:
    value: object
    type: Type


@dataclass(frozen=True)
class Call:
    op: Op
    args: tuple[Node, ...]

    @property
    def type(self) -> Type:
        return self.op.result


@dataclass(frozen=True)
class If:
    cond: Node
    then: Node
    orelse: Node
    type: Type


Node = Col | Lit | Call | If

TRUE = Lit(True, BOOL)
FALSE = Lit(False, BOOL)


def lit(value: object, type: Type | None = None) -> Lit:
    resolved = type or type_of(value)
    if resolved is None:
        raise TypeError(f"no IR type for literal {value!r}")
    return Lit(value, resolved)
