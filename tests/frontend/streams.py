from __future__ import annotations

import dis
from collections.abc import Callable, Mapping

import pytest

from udfx.frontend import trace
from udfx.frontend.decode import decoder
from udfx.frontend.decode.tables import resolve
from udfx.program import Program
from udfx.types import Type

Version = tuple[int, int]

VERSIONS = [(3, 12), (3, 13), (3, 14), (3, 15)]
VERSION_IDS = [f"{major}.{minor}" for major, minor in VERSIONS]


def traced(
    monkeypatch: pytest.MonkeyPatch,
    function: Callable[..., object],
    instructions: list[dis.Instruction],
    version: Version,
    schema: Mapping[str, Type],
) -> Program:
    numbered = tuple(item._replace(offset=2 * index) for index, item in enumerate(instructions))
    decoded = decoder.build(function.__code__, numbered, [], resolve(version))
    original = decoder.decode
    monkeypatch.setattr(
        "udfx.frontend.flow.frame.decode",
        lambda code: decoded if code is function.__code__ else original(code),
    )
    return trace(function, schema)
