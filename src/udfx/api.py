from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from udfx.backends import named
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
    backend: str

    @property
    def ok(self) -> bool:
        return self.expr is not None

    def __repr__(self) -> str:
        detail = (
            f"result={self.program.result}"
            if self.program is not None
            else f"unsupported={str(self.unsupported)!r}"
        )
        return f"Translation(ok={self.ok}, backend={self.backend!r}, {detail})"


def compile[E](
    function: Callable[..., object],
    schema: Mapping[str, Type],
    backend: Backend[E] | str = "polars",
) -> Translation[E]:
    target: Backend[E] = named(backend) if isinstance(backend, str) else backend
    try:
        program = trace(function, schema)
        return Translation(target.build(program), program, None, target.name)
    except Unsupported as reason:
        return Translation(None, None, reason, target.name)
