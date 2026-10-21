from __future__ import annotations

import dis
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import CodeType
from typing import NamedTuple

from udfx.errors import Unsupported
from udfx.frontend.constants import constant
from udfx.frontend.frozen import Snapshot
from udfx.frontend.logic import conjunction, fold, negate
from udfx.frontend.opcodes import normalize
from udfx.frontend.state import Raise, State, Step
from udfx.frontend.values import Item, Opaque, Value
from udfx.ir import FALSE, lit
from udfx.ops import Fault

JUMPS = frozenset(dis.opname[opcode] for opcode in (*dis.hasjrel, *dis.hasjabs))


class Protected(NamedTuple):
    start: int
    end: int
    target: int
    depth: int
    lasti: bool


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
    protected: tuple[Protected, ...]
    caller: tuple[Context, State] | None
    snapshot: Snapshot
    depth: int

    @classmethod
    def from_code(
        cls,
        code: CodeType,
        scope: Mapping[str, object],
        builtins: Mapping[str, object],
        caller: tuple[Context, State] | None,
        snapshot: Snapshot,
    ) -> Context:
        instructions = normalize(code)
        index_of = {instruction.offset: index for index, instruction in enumerate(instructions)}
        protected = tuple(
            Protected(entry.start, entry.end, entry.target, entry.depth, entry.lasti)
            for entry in dis.Bytecode(code).exception_entries  # pyright: ignore[reportAttributeAccessIssue]
        )
        depth = caller[0].depth + 1 if caller else 0
        return cls(
            code, scope, builtins, instructions, index_of, protected, caller, snapshot, depth
        )

    def frame(self, state: State, home: object) -> State:
        if self is home:
            return state
        if self.caller is None:
            raise Unsupported("helper escapes its defining function")
        context, outer = self.caller
        return context.frame(outer, home)

    def where(self, state: State) -> str:
        return f"{self.code.co_qualname}+{self.instructions[state.index].offset}"

    def target(self, instruction: dis.Instruction) -> int:
        if instruction.opname not in JUMPS or not isinstance(instruction.argval, int):
            raise Unsupported(f"{instruction.opname} has no jump target")
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
            when = fold(fault.when)
            failing = conjunction(path, when)
            if failing != FALSE:
                steps.append(self.raised(state.under(failing), fault.kind))
            path = conjunction(path, negate(when))
        return (state.under(path) if path != FALSE else None), steps

    def raised(self, state: State, kind: type[Exception]) -> Step:
        offset = self.instructions[state.index].offset
        for entry in self.protected:
            if entry.start <= offset < entry.end:
                kept = state.stack[: entry.depth]
                pushed = (Value(lit(offset)), Opaque(kind)) if entry.lasti else (Opaque(kind),)
                index = self.index_of[entry.target]
                return State(index, (*kept, *pushed), state.names, state.path)
        return Raise(state.path, kind)
