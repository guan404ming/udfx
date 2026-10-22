from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

from udfx.errors import Unsupported
from udfx.ir import Element
from udfx.ops import Op
from udfx.program import Program
from udfx.types import Type

ROOT = Path(__file__).resolve().parents[2]
PATH = Path(__file__).with_name(f"golden_{sys.version_info.major}_{sys.version_info.minor}.json")
RECORDING = bool(os.environ.get("UDFX_GOLDEN_RECORD"))
Tracer = Callable[[Callable[..., object], Mapping[str, Type]], Program]


def render(value: object, names: dict[str, int]) -> str:
    if isinstance(value, Op):
        return f"{value.name}{value.parameters}->{value.result}"
    if isinstance(value, Element):
        index = names.setdefault(value.name, len(names))
        return f"Element(e{index},{render(value.type, names)})"
    if is_dataclass(value) and not isinstance(value, type):
        parts = (render(getattr(value, field.name), names) for field in fields(value))
        return f"{type(value).__name__}({','.join(parts)})"
    if isinstance(value, tuple | list):
        return f"[{','.join(render(member, names) for member in value)}]"
    return repr(value)


def digest(program: Program) -> str:
    names: dict[str, int] = {}
    text = "|".join(
        (render(program.value, names), render(program.fault, names), str(program.row_mode))
    )
    return "p:" + hashlib.sha256(text.encode()).hexdigest()[:16]


def describe(error: Unsupported) -> str:
    return "u:" + re.sub(r"\+\d+", "+N", str(error))


def key(function: Callable[..., object], schema: Mapping[str, Type]) -> str:
    code = getattr(function, "__code__", None)
    if code is None:
        return f"{function!r}|{dict(schema)}"
    path = Path(code.co_filename)
    place = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name
    shape = hashlib.sha1(code.co_code).hexdigest()[:8]
    columns = ",".join(f"{name}:{kind}" for name, kind in schema.items())
    return f"{place}:{code.co_firstlineno}:{code.co_qualname}:{shape}|{columns}"


class Golden:
    def __init__(self) -> None:
        self.expected: dict[str, str] = {}
        self.seen: dict[str, str] = {}
        self.ambiguous: set[str] = set()
        if not RECORDING and PATH.exists():
            self.expected = json.loads(PATH.read_text())

    def knows(self, identifier: str) -> bool:
        return identifier in self.expected

    def observe(self, identifier: str, outcome: str) -> None:
        if RECORDING:
            if self.seen.setdefault(identifier, outcome) != outcome:
                self.ambiguous.add(identifier)
        elif identifier in self.expected and self.expected[identifier] != outcome:
            raise AssertionError(
                f"{identifier}: expected {self.expected[identifier]}, traced {outcome}"
            )

    def save(self) -> None:
        kept = {
            name: value for name, value in sorted(self.seen.items()) if name not in self.ambiguous
        }
        lines = ",\n".join(
            f"{json.dumps(name)}: {json.dumps(value)}" for name, value in kept.items()
        )
        PATH.write_text("{\n" + lines + "\n}\n")


GOLDEN = Golden()


def guarded(tracer: Tracer) -> Tracer:
    def traced(function: Callable[..., object], schema: Mapping[str, Type]) -> Program:
        identifier = key(function, schema)
        try:
            program = tracer(function, schema)
        except Unsupported as error:
            GOLDEN.observe(identifier, describe(error))
            raise
        GOLDEN.observe(identifier, digest(program))
        return program

    return traced
