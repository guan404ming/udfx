from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import CodeType
from typing import ClassVar

from udfx.frontend.decode.ops import Edge, Op
from udfx.frontend.decode.refusals import FEATURES


@dataclass(frozen=True)
class Jump(Op):
    edge: Edge


@dataclass(frozen=True)
class Branch(Op):
    condition: int
    then: Edge
    orelse: Edge


@dataclass(frozen=True)
class Return(Op):
    source: int


@dataclass(frozen=True)
class Fail(Op):
    exception: int


@dataclass(frozen=True)
class Refuse(Op):
    reason: str


@dataclass(frozen=True)
class ForEach(Op):
    reason: ClassVar[str] = FEATURES["FOR_ITER"]
    iterator: int
    item: int
    advanced: int
    dummy: int
    accumulator: int | None
    body: Edge
    done: Edge
    end: int


Exit = Jump | Branch | Return | Fail | Refuse | ForEach


@dataclass(frozen=True)
class Block:
    offset: int
    parameters: tuple[int, ...]
    ops: tuple[Op, ...]
    exit: Exit


@dataclass(frozen=True)
class Decoded:
    code: CodeType
    blocks: Mapping[int, Block]
    entry: int
    catches: bool
    orphans: Mapping[int, Block] = field(default_factory=dict)

    def loop(self) -> tuple[Block, ForEach]:
        loops = [
            (block, block.exit) for block in self.blocks.values() if isinstance(block.exit, ForEach)
        ]
        return min(loops, key=lambda loop: loop[0].offset)

    def span(self, low: int, high: int) -> tuple[Block, ...]:
        return tuple(block for _, block in sorted(self.blocks.items()) if low < block.offset < high)

    def operations(self, low: int, high: int | None = None) -> tuple[Op, ...]:
        return tuple(
            op
            for _, block in sorted({**self.orphans, **self.blocks}.items())
            for op in (*block.ops, block.exit)
            if op.offset >= low and (high is None or op.offset < high)
        )
