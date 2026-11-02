from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from udfx.aggregates import Aggregate, build_aggregate
from udfx.backends import named
from udfx.backends.base import Backend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.program import Program
from udfx.types import Type


@dataclass(frozen=True)
class Translation[E]:
    function: Callable[..., object]
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
    *,
    result: Type | None = None,
    series: bool = False,
) -> Translation[E]:
    target: Backend[E] = named(backend) if isinstance(backend, str) else backend
    try:
        program = trace(function, schema, result=result, pandas="scalar" if series else None)
        return Translation(function, target.build(program), program, None, target.name)
    except Unsupported as reason:
        return Translation(function, None, None, reason, target.name)


def aggregate[E](
    function: Callable[..., object],
    schema: Mapping[str, Type],
    backend: Backend[E] | str = "polars",
    *,
    result: Type | None = None,
) -> Translation[Aggregate[E]]:
    target: Backend[E] = named(backend) if isinstance(backend, str) else backend
    try:
        program = trace(function, schema, result=result, pandas="aggregate")
        return Translation(function, build_aggregate(target, program), program, None, target.name)
    except Unsupported as reason:
        return Translation(function, None, None, reason, target.name)
