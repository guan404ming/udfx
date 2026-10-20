from __future__ import annotations

import builtins
import dis
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from udfx.errors import Unsupported
from udfx.frontend.logic import conj, neg
from udfx.frontend.state import Raise, State, Step
from udfx.frontend.values import Item, Opaque, Value
from udfx.ir import FALSE, lit
from udfx.ops import Fault
from udfx.types import type_of

JUMPS = frozenset(dis.hasjrel) | frozenset(dis.hasjabs)


@dataclass(frozen=True)
class Context:
    function: Callable[..., object]
    instructions: tuple[dis.Instruction, ...]
    index_of: Mapping[int, int]

    @classmethod
    def of(cls, function: Callable[..., object]) -> Context:
        instructions = tuple(dis.get_instructions(function))
        index_of = {inst.offset: index for index, inst in enumerate(instructions)}
        return cls(function, instructions, index_of)

    def where(self, state: State) -> str:
        inst = self.instructions[state.index]
        return f"{self.function.__qualname__}+{inst.offset} {inst.opname}"

    def target(self, inst: dis.Instruction) -> int:
        if inst.opcode not in JUMPS or not isinstance(inst.argval, int):
            raise Unsupported(f"{inst.opname} has no jump target")
        return self.index_of[inst.argval]

    def constant(self, value: object) -> Item:
        return Value(lit(value)) if type_of(value) is not None else Opaque(value)

    def global_(self, name: str) -> Item:
        scope = getattr(self.function, "__globals__", {})
        if name in scope:
            return self.constant(scope[name])
        if hasattr(builtins, name):
            return Opaque(getattr(builtins, name))
        raise Unsupported(f"unknown global {name}")

    def guard(self, state: State, faults: Sequence[Fault]) -> tuple[State | None, list[Step]]:
        steps: list[Step] = []
        path = state.path
        for fault in faults:
            failing = conj(path, fault.when)
            if failing != FALSE:
                steps.append(self.raised(state.under(failing), fault.kind))
            path = conj(path, neg(fault.when))
        return (state.under(path) if path != FALSE else None), steps

    def raised(self, state: State, kind: type[Exception]) -> Step:
        return Raise(state.path, kind)
