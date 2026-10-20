from __future__ import annotations

from dataclasses import dataclass

from udfx.ir import Node


@dataclass(frozen=True)
class Value:
    node: Node


@dataclass(frozen=True)
class Method:
    name: str


@dataclass(frozen=True)
class Opaque:
    obj: object


@dataclass(frozen=True)
class Null:
    pass


NULL = Null()

Item = Value | Method | Opaque | Null
