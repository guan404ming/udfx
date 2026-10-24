from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from udfx.types import BOOL, Type, list_of, type_of

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
    arguments: tuple[Node, ...]

    @property
    def type(self) -> Type:
        return self.op.result


@dataclass(frozen=True)
class If:
    condition: Node
    then: Node
    orelse: Node
    type: Type


@dataclass(frozen=True)
class ListValue:
    items: tuple[Node, ...]
    type: Type


@dataclass(frozen=True)
class Element:
    name: str
    type: Type


@dataclass(frozen=True)
class Comprehension:
    source: Node
    element: Element
    value: Node
    condition: Node
    fault: Node
    requires_native: bool = False

    @property
    def type(self) -> Type:
        return list_of(self.value.type)


@dataclass(frozen=True)
class Guarded:
    value: Node
    exact: Node

    @property
    def type(self) -> Type:
        return self.value.type


Node = Col | Lit | Call | If | ListValue | Element | Comprehension | Guarded


def cached_hash(self: Any) -> int:
    found = self.__dict__.get("hashed")
    if found is None:
        found = hash(tuple(getattr(self, field.name) for field in fields(self)))
        object.__setattr__(self, "hashed", found)
    return found


def unhashed_state(self: Any) -> dict[str, object]:
    return {name: value for name, value in self.__dict__.items() if name != "hashed"}


for composite in (Call, If, ListValue, Comprehension, Guarded):
    composite.__hash__ = cached_hash
    composite.__getstate__ = unhashed_state

TRUE = Lit(True, BOOL)
FALSE = Lit(False, BOOL)


def lit(value: object, type: Type | None = None) -> Lit:
    resolved = type or type_of(value)
    if resolved is None:
        raise TypeError(f"no IR type for literal {value!r}")
    return Lit(value, resolved)


def children(node: Node) -> tuple[Node, ...]:
    match node:
        case Call():
            return node.arguments
        case If():
            return node.condition, node.then, node.orelse
        case Guarded():
            return node.value, node.exact
        case ListValue():
            return node.items
        case Comprehension():
            return node.source, node.value, node.condition, node.fault
        case _:
            return ()
