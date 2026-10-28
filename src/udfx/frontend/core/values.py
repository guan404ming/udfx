from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import CodeType, ModuleType

from udfx.errors import Unsupported
from udfx.ir import Col, Element, Lit, Node
from udfx.types import Kind

MAX_UNROLL = 64


def require_bounded(count: int) -> None:
    if count > MAX_UNROLL:
        raise Unsupported(f"loops and comprehensions exceed {MAX_UNROLL} unrolled items")


def bounded_iterator(items: Sequence[Item]) -> Iterator:
    require_bounded(len(items))
    return Iterator(tuple(items))


@dataclass(frozen=True)
class Value:
    node: Node
    container: type | None = None


@dataclass(frozen=True)
class Json:
    node: Node
    container: type | None = None


@dataclass(frozen=True)
class JsonType:
    node: Node


def wrapped(node: Node) -> Value | Json:
    return Json(node) if node.type.kind is Kind.JSON else Value(node)


@dataclass(frozen=True)
class Duration:
    node: Node


@dataclass(frozen=True)
class Match:
    node: Node
    source: Node
    pattern: str
    flags: int
    mode: str
    position: Node


@dataclass(frozen=True)
class Row:
    columns: Mapping[str, Value]

    def column(self, name: str | int) -> Value:
        if isinstance(name, int):
            return tuple(self.columns.values())[name]
        if name not in self.columns:
            raise Unsupported(f"unknown row column {name!r}")
        return self.columns[name]


@dataclass(frozen=True)
class Method:
    name: str


@dataclass(frozen=True)
class Opaque:
    value: object


@dataclass(frozen=True)
class RuntimeType:
    node: Node


@dataclass(frozen=True)
class Tuple:
    items: tuple[Item, ...]


@dataclass(frozen=True)
class Iterator:
    items: tuple[Item, ...]
    position: int = 0
    source: Node | None = None
    element: Element | None = None
    value: Item | None = None
    requires_native: bool = False
    json: bool = False


@dataclass(frozen=True)
class Generator:
    node: Node


@dataclass(frozen=True)
class Set:
    node: Node
    dynamic: bool = False


@dataclass(frozen=True)
class Counter:
    node: Node


@dataclass(frozen=True)
class CounterKey:
    owner: Counter


@dataclass(frozen=True)
class Slice:
    bounds: tuple[Node, ...]


@dataclass(frozen=True)
class Null:
    pass


@dataclass(frozen=True)
class Function:
    code: CodeType
    cells: tuple[str, ...]
    home: object = field(repr=False)
    defaults: tuple[Item, ...] = ()
    keyword_defaults: Mapping[str, Item] = field(default_factory=dict)


@dataclass(frozen=True)
class Cell:
    name: str


NULL = Null()

Item = (
    Value
    | Json
    | JsonType
    | Match
    | Duration
    | Row
    | Method
    | Opaque
    | RuntimeType
    | Tuple
    | Iterator
    | Generator
    | Set
    | Counter
    | CounterKey
    | Slice
    | Null
    | Function
)


def named(value: object) -> str:
    qualified_name = getattr(value, "__qualname__", None)
    if isinstance(qualified_name, str):
        module = getattr(value, "__module__", None)
        if module is None or getattr(builtins, qualified_name, None) is value:
            return qualified_name
        return f"{module}.{qualified_name}"
    if isinstance(value, ModuleType):
        return f"module {value.__name__}"
    return f"a {type(value).__qualname__} object"


def describe(item: Item) -> str:
    match item:
        case Value(node=Col(name=name)):
            return f"column {name}"
        case Value(node=Lit(value=value)):
            return repr(value)
        case Value(node=node):
            return f"a {node.type} value"
        case Match():
            return "a regex match (use a truth test, a None test, or guarded group(0))"
        case Row():
            return "a row"
        case Method(name=name):
            return f"method .{name}"
        case Opaque(value=value):
            return named(value)
        case RuntimeType(node=node):
            return f"type of a {node.type} value"
        case Iterator():
            return "an iterator"
        case (
            Tuple()
            | Generator()
            | Set()
            | Slice()
            | Duration()
            | Json()
            | JsonType()
            | Counter()
            | CounterKey()
        ):
            return f"a {type(item).__name__.lower()}"
        case Function(code=code):
            return f"helper {code.co_qualname}"
        case Null():
            return "nothing"
