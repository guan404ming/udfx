from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from udfx.backends.base import Backend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.program import Program
from udfx.types import Type


@dataclass(frozen=True)
class Translation[E]:
    expr: E | None
    program: Program | None
    unsupported: Unsupported | None

    @property
    def ok(self) -> bool:
        return self.expr is not None


def backend_named(name: str) -> Backend[Any]:
    module = importlib.import_module(f"udfx.backends.{name}")
    backend: type[Backend[Any]] = module.BACKEND
    return backend()


def compile[E](
    function: Callable[..., object],
    schema: Mapping[str, Type],
    backend: Backend[E] | str = "polars",
) -> Translation[E]:
    target: Backend[E] = backend_named(backend) if isinstance(backend, str) else backend
    try:
        program = trace(function, schema)
        return Translation(target.build(program), program, None)
    except Unsupported as reason:
        return Translation(None, None, reason)
