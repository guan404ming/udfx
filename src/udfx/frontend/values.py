from __future__ import annotations

from dataclasses import dataclass, field
from types import CodeType

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


@dataclass(frozen=True)
class Function:
    code: CodeType
    cells: tuple[str, ...]
    home: object = field(repr=False)


@dataclass(frozen=True)
class Cell:
    name: str


NULL = Null()

Item = Value | Method | Opaque | Null | Function
