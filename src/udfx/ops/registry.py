from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from udfx.ir import Call, Node
from udfx.types import Kind, Type


@dataclass(frozen=True)
class Fault:
    kind: type[Exception]
    when: Node


@dataclass(frozen=True)
class OpKey:
    name: str
    kinds: tuple[Kind, ...]

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.kinds)})"


def no_faults(*_: Node) -> tuple[Fault, ...]:
    return ()


@dataclass(frozen=True, eq=False)
class Op:
    name: str
    params: tuple[Type, ...]
    result: Type
    reference: Callable[..., object]
    faults: Callable[..., tuple[Fault, ...]] = field(default=no_faults)

    @property
    def key(self) -> OpKey:
        return OpKey(self.name, tuple(param.kind for param in self.params))

    def __str__(self) -> str:
        return f"{self.key} -> {self.result}"


_OPS: dict[OpKey, Op] = {}


def register(op: Op) -> Op:
    if op.key in _OPS:
        raise ValueError(f"duplicate op {op.key}")
    _OPS[op.key] = op
    return op


def all_ops() -> tuple[Op, ...]:
    return tuple(_OPS.values())


def resolve(name: str, kinds: tuple[Kind, ...]) -> Op | None:
    return _OPS.get(OpKey(name, kinds))


def apply(name: str, *args: Node) -> Call:
    kinds = tuple(arg.type.kind for arg in args)
    op = resolve(name, kinds)
    if op is None:
        raise LookupError(f"no op {OpKey(name, kinds)}")
    return Call(op, args)


@contextmanager
def scoped() -> Iterator[None]:
    saved = dict(_OPS)
    try:
        yield
    finally:
        _OPS.clear()
        _OPS.update(saved)
