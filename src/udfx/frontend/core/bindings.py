from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from udfx.errors import Unsupported
from udfx.frontend.core.values import Generator, Item, Iterator, Row
from udfx.ir import Node


@dataclass(frozen=True)
class Bindings:
    names: Mapping[str, Item]
    path: Node

    def store(self, name: str, item: Item) -> Bindings:
        if isinstance(item, Generator | Iterator):
            raise Unsupported("generator must be consumed immediately")
        if isinstance(item, Row):
            raise Unsupported("assignment of a row")
        return replace(self, names={**self.names, name: item})

    def load(self, name: str) -> Item:
        if name not in self.names:
            raise Unsupported(f"unbound local {name}")
        return self.names[name]

    def delete(self, name: str) -> Bindings:
        return replace(self, names={key: item for key, item in self.names.items() if key != name})

    def under(self, path: Node) -> Bindings:
        return replace(self, path=path)
