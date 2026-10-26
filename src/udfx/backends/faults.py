from __future__ import annotations

import builtins
import json
import re
from datetime import datetime
from typing import Any
from weakref import WeakKeyDictionary

from udfx.program import Program
from udfx.types import Kind, Type

MARKER = "udfx fault "
PATTERN = re.compile(r"udfx fault ([0-9a-f]+):([0-9A-Fa-f]+)")
FALLBACK_MARKER = "udfx raised "


def encode_fallback_error(error: Exception) -> str:
    detail = str(error)
    if isinstance(error, json.JSONDecodeError):
        detail = json.dumps([error.msg, error.doc, error.pos]).encode().hex()
    return f"{FALLBACK_MARKER}{type(error).__name__}: {detail}"


def decode_fallback_error(text: str) -> Exception | None:
    _, found, detail = text.partition(FALLBACK_MARKER)
    name, _, message = detail.partition(": ")
    if found and name.strip() == "JSONDecodeError":
        encoded = re.match(r"[0-9a-f]+", message)
        if encoded is not None:
            try:
                return json.JSONDecodeError(*json.loads(bytes.fromhex(encoded.group())))
            except (ValueError, TypeError):
                return None
    kind = getattr(builtins, name.strip(), None)
    if not found or not isinstance(kind, type) or not issubclass(kind, Exception):
        return None
    try:
        return kind(message.partition("\n")[0])
    except TypeError:
        return None


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
