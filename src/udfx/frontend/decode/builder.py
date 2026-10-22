from __future__ import annotations

import dis
from collections.abc import Callable
from dataclasses import replace
from types import CodeType
from typing import Any

from udfx.frontend.decode.graph import Exit
from udfx.frontend.decode.ops import (
    NULL,
    Boolean,
    Constant,
    DecodeError,
    Edge,
    Handler,
    Not,
    Op,
)

Rule = Callable[["Builder", dis.Instruction], None]


class Builder:
    def __init__(self, code: CodeType, instructions: tuple[dis.Instruction, ...]) -> None:
        self.code = code
        self.instructions = instructions
        self.index_of = {
            instruction.offset: index for index, instruction in enumerate(instructions)
        }
        self.cells = frozenset((*code.co_cellvars, *code.co_freevars))
        self.counter = 0
        self.claims: dict[int, tuple[int, ...]] = {}
        self.pending: list[int] = []
        self.stack: list[int] = []
        self.ops: list[Op] = []
        self.exit: Exit | None = None
        self.constants: dict[int, object] = {}
        self.nulls = {NULL}
        self.boolean: set[int] = set()
        self.keywords: tuple[str, ...] = ()
        self.offset = 0
        self.handler: Handler | None = None
        self.index = 0
        self.carried = 0
        self.lenient = False
        self.visited: set[int] = set()

    def take(self) -> int:
        weight, self.carried = self.carried, 0
        return weight

    def new(self) -> int:
        self.counter += 1
        return self.counter

    def push(self, *identifiers: int) -> None:
        self.stack.extend(identifiers)

    def pop(self) -> int:
        return self.pop_many(1)[0]

    def reach(self, depth: int) -> None:
        missing = depth - len(self.stack)
        if missing > 0 and not self.lenient:
            raise DecodeError(f"stack underflow at {self.offset}")
        if missing > 0:
            self.stack[:0] = [self.new() for _ in range(missing)]

    def pop_many(self, count: int) -> tuple[int, ...]:
        self.reach(count)
        cut = len(self.stack) - count
        popped = tuple(self.stack[cut:])
        del self.stack[cut:]
        return popped

    def emit[Result: Op](
        self, kind: Callable[..., Result], *arguments: Any, **fields: Any
    ) -> Result:
        operation = kind(
            *arguments, offset=self.offset, handler=self.handler, weight=self.take(), **fields
        )
        self.ops.append(operation)
        return operation

    def produce(self, kind: Callable[..., Op], *arguments: Any, **fields: Any) -> int:
        result = self.new()
        self.emit(kind, result, *arguments, **fields)
        return result

    def constant(self, value: object) -> int:
        result = self.produce(Constant, value)
        self.constants[result] = value
        return result

    def truth(self, source: int) -> int:
        if source in self.boolean:
            return source
        result = self.produce(Boolean, source)
        self.boolean.add(result)
        return result

    def negation(self, source: int) -> int:
        result = self.produce(Not, source)
        self.boolean.add(result)
        return result

    def claim(self, target: int, count: int) -> None:
        if self.lenient:
            if target not in self.claims:
                self.claims[target] = ()
                self.pending.append(target)
            return
        if target not in self.claims:
            self.claims[target] = tuple(self.new() for _ in range(count))
            self.pending.append(target)
        elif len(self.claims[target]) != count:
            raise DecodeError(
                f"block {target} reached with {count} values, expected {len(self.claims[target])}"
            )

    def edge(self, target: int) -> Edge:
        self.claim(target, len(self.stack))
        self.nulls.update(
            parameter
            for parameter, source in zip(self.claims[target], self.stack, strict=False)
            if source in self.nulls
        )
        return Edge(target, tuple(self.stack))

    def finish(self, kind: Callable[..., Exit], *arguments: Any, **fields: Any) -> None:
        self.exit = kind(
            *arguments, offset=self.offset, handler=self.handler, weight=self.take(), **fields
        )

    def retract(self, result: int) -> None:
        if self.ops and getattr(self.ops[-1], "result", None) == result:
            self.carried += self.ops.pop().weight
            self.constants.pop(result, None)

    def replace_operation(self, result: int, **fields: Any) -> None:
        for position in range(len(self.ops) - 1, -1, -1):
            if getattr(self.ops[position], "result", None) == result:
                self.ops[position] = replace(self.ops[position], **fields)
                return
        raise DecodeError(f"no operation produces {result}")

    def slots(self, first: int, second: int) -> tuple[int, int | None]:
        if first in self.nulls:
            return second, None
        if second in self.nulls:
            return first, None
        return first, second
