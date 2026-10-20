from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from udfx.ir import Call, Node
from udfx.types import Kind, Type, accepts


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
    parameters: tuple[Type, ...]
    result: Type
    reference: Callable[..., object]
    faults: Callable[..., tuple[Fault, ...]] = field(default=no_faults)

    @property
    def key(self) -> OpKey:
        return OpKey(self.name, tuple(parameter.kind for parameter in self.parameters))

    def __str__(self) -> str:
        return f"{self.name}({', '.join(map(str, self.parameters))}) -> {self.result}"


_OPS: dict[tuple[str, tuple[Type, ...]], Op] = {}


def signature(types: Sequence[Type]) -> tuple[Type, ...]:
    return tuple(type.strict() for type in types)


def register(op: Op) -> Op:
    key = (op.name, signature(op.parameters))
    if key in _OPS:
        raise ValueError(f"duplicate op {op}")
    _OPS[key] = op
    return op


def all_ops() -> tuple[Op, ...]:
    return tuple(_OPS.values())


def matching(name: str, types: Sequence[Type]) -> list[Op]:
    found = [
        op
        for op in _OPS.values()
        if op.name == name
        and len(op.parameters) == len(types)
        and all(map(accepts, op.parameters, types))
    ]
    return sorted(found, key=lambda op: signature(op.parameters) != signature(types))


def resolve(name: str, types: Sequence[Type]) -> Op | None:
    found = matching(name, types)
    return found[0] if found else None


def apply(name: str, *arguments: Node) -> Call:
    op = resolve(name, [argument.type for argument in arguments])
    if op is None:
        raise LookupError(
            f"no op {name}({', '.join(str(argument.type) for argument in arguments)})"
        )
    return Call(op, arguments)


@contextmanager
def scoped() -> Generator[None]:
    saved = dict(_OPS)
    try:
        yield
    finally:
        _OPS.clear()
        _OPS.update(saved)
