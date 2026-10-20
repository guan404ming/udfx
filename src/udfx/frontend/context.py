from __future__ import annotations

import dis
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import CodeType

from udfx.errors import Unsupported
from udfx.frontend.logic import conj, neg
from udfx.frontend.state import Raise, State, Step
from udfx.frontend.values import Item, Opaque, Value
from udfx.ir import FALSE, lit
from udfx.ops import Fault

JUMPS = frozenset(dis.hasjrel) | frozenset(dis.hasjabs)

SCALARS = (str, int, float, bool)


def constant(value: object) -> Item:
    return Value(lit(value)) if value is None or type(value) in SCALARS else Opaque(value)


def parameters(code: CodeType) -> list[str]:
    if code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS) or code.co_kwonlyargcount:
        raise Unsupported(f"{code.co_qualname} takes variable or keyword-only parameters")
    return list(code.co_varnames[: code.co_argcount])


@dataclass(frozen=True)
class Context:
    code: CodeType
    scope: Mapping[str, object]
    builtins: Mapping[str, object]
    instructions: tuple[dis.Instruction, ...]
    index_of: Mapping[int, int]

    @classmethod
    def of(
        cls, code: CodeType, scope: Mapping[str, object], builtins: Mapping[str, object]
    ) -> Context:
        instructions = tuple(dis.get_instructions(code))
        index_of = {instruction.offset: index for index, instruction in enumerate(instructions)}
        return cls(code, scope, builtins, instructions, index_of)

    def where(self, state: State) -> str:
        instruction = self.instructions[state.index]
        return f"{self.code.co_qualname}+{instruction.offset} {instruction.opname}"

    def target(self, instruction: dis.Instruction) -> int:
        if instruction.opcode not in JUMPS or not isinstance(instruction.argval, int):
            raise Unsupported(f"{instruction.opname} has no jump target")
        if instruction.argval <= instruction.offset:
            raise Unsupported("loops")
        return self.index_of[instruction.argval]

    def global_(self, name: str) -> Item:
        if name in self.scope:
            return constant(self.scope[name])
        if name in self.builtins:
            return Opaque(self.builtins[name])
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
