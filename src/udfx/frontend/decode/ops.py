from __future__ import annotations

from dataclasses import dataclass
from types import CodeType
from typing import ClassVar

from udfx.frontend.decode.refusals import FEATURES

NULL = -1


class DecodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Edge:
    target: int
    arguments: tuple[int, ...]


@dataclass(frozen=True)
class Handler:
    edge: Edge
    exception: int


@dataclass(frozen=True, kw_only=True)
class Op:
    offset: int
    handler: Handler | None = None
    weight: int = 0


@dataclass(frozen=True)
class Constant(Op):
    result: int
    value: object


@dataclass(frozen=True)
class Local(Op):
    result: int
    name: str
    clear: bool = False
    cell: bool = False


@dataclass(frozen=True)
class Cell(Op):
    result: int
    name: str


@dataclass(frozen=True)
class Store(Op):
    name: str
    source: int
    cell: bool = False


@dataclass(frozen=True)
class Delete(Op):
    name: str


@dataclass(frozen=True)
class Global(Op):
    result: int
    name: str


@dataclass(frozen=True)
class Placeholder(Op):
    result: int


@dataclass(frozen=True)
class Attribute(Op):
    results: tuple[int, ...]
    owner: int
    name: str
    method: bool


@dataclass(frozen=True)
class Call(Op):
    result: int
    callee: int
    arguments: tuple[int, ...]
    keywords: tuple[tuple[str, int], ...]
    receiver: int | None


@dataclass(frozen=True)
class UnpackedCall(Op):
    result: int
    callee: int
    receiver: int | None
    arguments: int
    keywords: int | None


@dataclass(frozen=True)
class BinaryOp(Op):
    result: int
    symbol: str
    left: int
    right: int


@dataclass(frozen=True)
class Compare(Op):
    result: int
    symbol: str
    left: int
    right: int


@dataclass(frozen=True)
class Negate(Op):
    result: int
    source: int


@dataclass(frozen=True)
class Not(Op):
    result: int
    source: int


@dataclass(frozen=True)
class Boolean(Op):
    result: int
    source: int


@dataclass(frozen=True)
class Subscript(Op):
    result: int
    container: int
    index: int


@dataclass(frozen=True)
class Slice(Op):
    result: int
    bounds: tuple[int, ...]


@dataclass(frozen=True)
class Cut(Op):
    result: int
    container: int
    start: int
    stop: int


@dataclass(frozen=True)
class Build(Op):
    result: int
    kind: str
    items: tuple[int, ...]


@dataclass(frozen=True)
class Extend(Op):
    result: int
    kind: str
    container: int
    source: int


@dataclass(frozen=True)
class Unpack(Op):
    results: tuple[int, ...]
    source: int
    before: int
    after: int
    starred: bool


@dataclass(frozen=True)
class Format(Op):
    result: int
    source: int
    conversion: str | None
    specification: int | None


@dataclass(frozen=True)
class MakeFunction(Op):
    result: int
    code: CodeType
    closure: int | None = None
    defaults: int | None = None
    keyword_defaults: int | None = None


@dataclass(frozen=True)
class ExceptionMatch(Op):
    result: int
    exception: int
    classes: int


@dataclass(frozen=True)
class Iterate(Op):
    result: int
    source: int


@dataclass(frozen=True)
class Merge(Op):
    source: int


@dataclass(frozen=True)
class Append(Op):
    reason: ClassVar[str] = FEATURES["LIST_APPEND"]
    value: int


@dataclass(frozen=True)
class Yield(Op):
    reason: ClassVar[str] = FEATURES["YIELD_VALUE"]
    value: int
    result: int


@dataclass(frozen=True)
class GeneratorStart(Op):
    reason: ClassVar[str] = FEATURES["RETURN_GENERATOR"]
