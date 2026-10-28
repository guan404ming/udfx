from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from typing import TYPE_CHECKING

from udfx.backends.diagnostics import capture_lowering
from udfx.backends.lowering import lower_if
from udfx.backends.lowering_context import assuming_exact
from udfx.backends.rules import Lowered, Scope, fold_predicate
from udfx.ir import FALSE, TRUE, If, Lit, Node
from udfx.program import Program
from udfx.semantics.assumptions import assuming

if TYPE_CHECKING:
    from udfx.backends.base import Backend


class ResultLowering[E]:
    def __init__(self, backend: Backend[E], program: Program) -> None:
        self.backend = backend
        self.program = program
        self.raised: E | None = None
        self.native_errors = False

    def lower(self, node: Node, elements: Scope[E]) -> Lowered[E]:
        lowered = (
            lower_if(self.backend, node, elements, self.lower)
            if isinstance(node, If) and not (elements and node in elements)
            else self.backend.lower(node, elements)
        )
        return self.finish(lowered)

    def finish(self, lowered: Lowered[E]) -> Lowered[E]:
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


class FallbackLowering[E](ResultLowering[E]):
    def __init__(self, backend: Backend[E], program: Program) -> None:
        super().__init__(backend, program)
        self.pending: E | None = None

    def fallback(self) -> E:
        if self.pending is None:
            self.pending = self.backend.fallback(
                self.backend.literal(Lit(None, self.program.result.or_none())),
                self.backend.literal(FALSE),
                self.program,
            )
        return self.pending

    def lower(self, node: Node, elements: Scope[E]) -> Lowered[E]:
        if not isinstance(node, If) or (elements and node in elements):
            return super().lower(node, elements)
        backend = self.backend
        predicate = fold_predicate(node.condition, backend.rules.rewrite_call)
        if backend.exact(backend.lower(predicate, elements)) is None:
            return super().lower(node, elements)
        cases: list[tuple[E, E]] = []
        with ExitStack() as scope:
            while isinstance(node, If) and not (elements and node in elements):
                predicate = fold_predicate(node.condition, backend.rules.rewrite_call)
                if predicate in (TRUE, FALSE):
                    node = node.then if predicate == TRUE else node.orelse
                    continue
                condition = backend.lower(predicate, elements)
                exact = backend.exact(condition)
                if exact is not None:
                    failed = backend.select(exact, backend.literal(FALSE), backend.literal(TRUE))
                    cases.append((failed, self.fallback()))
                inverted = isinstance(node.then, If) and isinstance(node.orelse, Lit)
                with assuming(predicate, not inverted), assuming_exact(predicate):
                    output = self.lower(node.orelse if inverted else node.then, elements)
                check = (
                    backend.select(condition.expr, backend.literal(FALSE), backend.literal(TRUE))
                    if inverted
                    else condition.expr
                )
                cases.append((check, output.expr))
                scope.enter_context(assuming(predicate, inverted))
                scope.enter_context(assuming_exact(predicate))
                node = node.then if inverted else node.orelse
            default = self.lower(node, elements)
        return Lowered(backend.select_many(cases, default.expr)) if cases else default

    def finish(self, lowered: Lowered[E]) -> Lowered[E]:
        lowered = super().finish(lowered)
        exact = self.backend.exact(lowered)
        return Lowered(
            lowered.expr
            if exact is None
            else self.backend.select(exact, lowered.expr, self.fallback())
        )


def fallback_result[E](
    value: Lowered[E],
    results: FallbackLowering[E],
    lower: Callable[[ResultLowering[E]], Lowered[E]],
) -> Lowered[E]:
    if value.exact is None or not results.program.deterministic:
        return value
    with capture_lowering():
        return lower(results)
