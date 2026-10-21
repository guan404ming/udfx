from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from udfx.errors import Unsupported
from udfx.frontend.values import Item, Row
from udfx.ir import Node


@dataclass(frozen=True)
class State:
    index: int
    stack: tuple[Item, ...]
    names: Mapping[str, Item]
    path: Node

    def push(self, *items: Item) -> State:
        return replace(self, stack=(*self.stack, *items))

    def pop(self, count: int = 1) -> tuple[tuple[Item, ...], State]:
        if count > len(self.stack):
            raise IndexError("stack underflow")
        cut = len(self.stack) - count
        return self.stack[cut:], replace(self, stack=self.stack[:cut])

    def store(self, name: str, item: Item) -> State:
        if isinstance(item, Row):
            raise Unsupported("assignment of a row")
        return replace(self, names={**self.names, name: item})

    def load(self, name: str) -> Item:
        if name not in self.names:
            raise Unsupported(f"unbound local {name}")
        return self.names[name]

    def delete(self, name: str) -> State:
        return replace(self, names={key: item for key, item in self.names.items() if key != name})

    def goto(self, index: int) -> State:
        return replace(self, index=index)

    def advance(self) -> State:
        return replace(self, index=self.index + 1)

    def under(self, path: Node) -> State:
        return replace(self, path=path)


@dataclass(frozen=True)
class Return:
    path: Node
    value: Node


@dataclass(frozen=True)
class Raise:
    path: Node
    kind: type[Exception]


Outcome = Return | Raise
Step = State | Outcome
