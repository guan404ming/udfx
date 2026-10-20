from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from types import CodeType, ModuleType

from udfx.ir import Col, Lit, Node


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


def named(obj: object) -> str:
    qualname = getattr(obj, "__qualname__", None)
    if isinstance(qualname, str):
        module = getattr(obj, "__module__", None)
        if module is None or getattr(builtins, qualname, None) is obj:
            return qualname
        return f"{module}.{qualname}"
    if isinstance(obj, ModuleType):
        return f"module {obj.__name__}"
    return f"a {type(obj).__qualname__} object"


def describe(item: Item) -> str:
    match item:
        case Value(node=Col(name=name)):
            return f"column {name}"
        case Value(node=Lit(value=value)):
            return repr(value)
        case Value(node=node):
            return f"a {node.type} value"
        case Method(name=name):
            return f"method .{name}"
        case Opaque(obj=obj):
            return named(obj)
        case Function(code=code):
            return f"helper {code.co_qualname}"
        case Null():
            return "nothing"
