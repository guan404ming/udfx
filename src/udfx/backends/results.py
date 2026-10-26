from __future__ import annotations

from typing import TYPE_CHECKING

from udfx.backends.lowering import lower_if
from udfx.backends.rules import Lowered, Scope
from udfx.ir import If, Node
from udfx.program import Program

if TYPE_CHECKING:
    from udfx.backends.base import Backend


class ResultLowering[E]:
    def __init__(self, backend: Backend[E], program: Program) -> None:
        self.backend = backend
        self.program = program
        self.raised: E | None = None
        self.native_errors = False

    def lower(self, node: Node, elements: Scope[E]) -> Lowered[E]:
        if isinstance(node, If) and not (elements and node in elements):
            return lower_if(self.backend, node, elements, self.lower)
        lowered = self.backend.lower(node, elements)
        if lowered.overflow is None:
            return lowered
        if self.raised is None:
            self.raised = self.backend.native_raise(self.program)
        if self.raised is None:
            return lowered
        self.native_errors = True
        condition = lowered.overflow
        if lowered.exact is not None:
            condition = self.backend.both(condition, lowered.exact)
        return Lowered(self.backend.select(condition, self.raised, lowered.expr), lowered.exact)
