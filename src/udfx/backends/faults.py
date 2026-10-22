from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any
from weakref import WeakKeyDictionary

from udfx.program import Program
from udfx.types import Kind, Type

MARKER = "udfx fault "
PATTERN = re.compile(r"udfx fault ([0-9a-f]+):([0-9A-Fa-f]+)")


def restore(value: Any, type: Type) -> object:
    if value is None:
        return None
    if type.kind is Kind.FLOAT:
        return float(value)
    if type.kind is Kind.DATETIME:
        return datetime.fromisoformat(value)
    if type.kind is Kind.LIST and type.item is not None:
        return [restore(item, type.item) for item in value]
    return value


class NativeFaults:
    def __init__(self) -> None:
        self.programs: dict[str, Program] = {}

    def marker(self, program: Program) -> str:
        key = f"{id(program):x}"
        self.programs[key] = program
        return f"{MARKER}{key}:"

    def decode(self, message: str) -> Exception | None:
        found = PATTERN.search(message)
        if found is None:
            return None
        key, encoded = found.groups()
        program = self.programs.get(key)
        if program is None:
            return None
        try:
            packed = json.loads(bytes.fromhex(encoded))
            row = [
                restore(packed.get(str(index)), type)
                for index, type in enumerate(program.schema.values())
            ]
        except (ValueError, TypeError, AttributeError):
            return None
        try:
            program.run([row])
        except Exception as error:
            return error
        return None


REGISTRIES: WeakKeyDictionary[object, NativeFaults] = WeakKeyDictionary()


def faults_for(owner: object) -> NativeFaults:
    return REGISTRIES.setdefault(owner, NativeFaults())
