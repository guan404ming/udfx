from __future__ import annotations

import dis
import inspect
import sys
from functools import cache
from types import CodeType
from typing import Any

from udfx.frontend.decode.builder import Builder, Rule
from udfx.frontend.decode.graph import Block, Decoded, Jump, Refuse
from udfx.frontend.decode.ops import Constant, DecodeError, Edge, Handler
from udfx.frontend.decode.rules.loads import refuse
from udfx.frontend.decode.tables import resolve

VERSION = (sys.version_info.major, sys.version_info.minor)
JUMPS = frozenset((*dis.hasjrel, *dis.hasjabs))
SPLITS = frozenset(
    {
        "POP_JUMP_IF_FALSE",
        "POP_JUMP_IF_TRUE",
        "POP_JUMP_IF_NONE",
        "POP_JUMP_IF_NOT_NONE",
        "FOR_ITER",
    }
)
CLEANUP = frozenset({"SWAP", "POP_TOP", "STORE_FAST", "STORE_FAST_STORE_FAST"})
SUSPENDING = inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR


def is_cleanup(builder: Builder, target: int) -> bool:
    names = [part.opname for part in builder.instructions[builder.index_of[target] :]]
    if names[:2] == ["CALL_INTRINSIC_1", "RERAISE"]:
        return True
    length = 0
    while names[length] in CLEANUP:
        length += 1
    return names[0] == "SWAP" and names[length] == "RERAISE"


def attach_handler(builder: Builder, entries: list[Any]) -> None:
    for entry in entries:
        if entry.start <= builder.offset < entry.end:
            if is_cleanup(builder, entry.target):
                return
            depth, lasti, target = entry.depth, entry.lasti, entry.target
            if depth > len(builder.stack):
                raise DecodeError(f"handler depth exceeds the stack at {builder.offset}")
            builder.claim(target, depth + int(lasti) + 1)
            arguments = tuple(builder.stack[:depth])
            if lasti:
                position = builder.new()
                builder.ops.append(Constant(position, builder.offset, offset=builder.offset))
                arguments += (position,)
            exception = builder.new()
            builder.handler = Handler(Edge(target, (*arguments, exception)), exception)
            return


def run_block(
    builder: Builder,
    offset: int,
    leaders: set[int],
    entries: list[Any],
    table: dict[str, Rule],
) -> Block:
    builder.stack = list(builder.claims[offset])
    builder.ops = []
    builder.exit = None
    builder.boolean = set()
    builder.keywords = ()
    builder.carried = 0
    start = builder.index_of[offset]
    index = start
    while builder.exit is None:
        instruction = builder.instructions[index]
        builder.index = index
        builder.offset = instruction.offset
        builder.handler = None
        builder.visited.add(instruction.offset)
        if index != start and instruction.offset in leaders:
            builder.finish(Jump, builder.edge(instruction.offset))
            break
        builder.carried += 1
        if not builder.lenient:
            attach_handler(builder, entries)
        table.get(instruction.opname, refuse)(builder, instruction)
        index += 1
    if builder.exit is None:
        raise DecodeError(f"block {offset} has no exit")
    return Block(offset, builder.claims[offset], tuple(builder.ops), builder.exit)


def run_blocks(
    builder: Builder, leaders: set[int], entries: list[Any], table: dict[str, Rule]
) -> dict[int, Block]:
    blocks: dict[int, Block] = {}
    while builder.pending:
        offset = builder.pending.pop()
        if offset not in blocks:
            blocks[offset] = run_block(builder, offset, leaders, entries, table)
    return blocks


def build(
    code: CodeType,
    instructions: tuple[dis.Instruction, ...],
    entries: list[Any],
    table: dict[str, Rule],
) -> Decoded:
    first = instructions[0].offset
    if code.co_flags & SUSPENDING:
        start = next(part for part in instructions if part.opname == "RETURN_GENERATOR")
        refusal = Refuse("generators", offset=start.offset)
        return Decoded(code, {first: Block(first, (), (), refusal)}, first, False)
    leaders = {entry.target for entry in entries}
    for index, instruction in enumerate(instructions):
        if instruction.opcode in JUMPS and isinstance(instruction.argval, int):
            leaders.add(instruction.argval)
        if instruction.opname in SPLITS:
            leaders.add(instructions[index + 1].offset)
    builder = Builder(code, instructions)
    builder.claim(first, 0)
    blocks = run_blocks(builder, leaders, entries, table)
    builder.lenient = True
    orphans: dict[int, Block] = {}
    for instruction in instructions:
        if instruction.offset not in builder.visited:
            builder.claims[instruction.offset] = ()
            builder.pending.append(instruction.offset)
            orphans.update(run_blocks(builder, leaders, entries, table))
    catches = any(part.opname == "PUSH_EXC_INFO" for part in instructions)
    return Decoded(code, blocks, first, catches, orphans)


@cache
def decode(code: CodeType) -> Decoded:
    entries = list(dis.Bytecode(code).exception_entries)  # pyright: ignore[reportAttributeAccessIssue]
    return build(code, tuple(dis.get_instructions(code)), entries, resolve(VERSION))
