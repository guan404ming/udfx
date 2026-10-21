from __future__ import annotations

import builtins
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import CodeType, ModuleType

from udfx.errors import Unsupported
from udfx.ir import Col, Lit, Node


@dataclass(frozen=True)
class Value:
    node: Node


@dataclass(frozen=True)
class Row:
    columns: Mapping[str, Value]

    def column(self, name: str) -> Value:
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

Item = Value | Row | Method | Opaque | Null | Function


def named(value: object) -> str:
    qualname = getattr(value, "__qualname__", None)
    if isinstance(qualname, str):
        module = getattr(value, "__module__", None)
        if module is None or getattr(builtins, qualname, None) is value:
            return qualname
        return f"{module}.{qualname}"
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
        case Row():
            return "a row"
        case Method(name=name):
            return f"method .{name}"
        case Opaque(value=value):
            return named(value)
        case Function(code=code):
            return f"helper {code.co_qualname}"
        case Null():
            return "nothing"
