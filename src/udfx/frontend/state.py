from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from udfx.frontend.values import Item
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

    def top(self) -> Item:
        return self.stack[-1]

    def store(self, name: str, item: Item) -> State:
        return replace(self, names={**self.names, name: item})

    def load(self, name: str) -> Item:
        return self.names[name]

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
