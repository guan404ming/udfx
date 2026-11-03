from __future__ import annotations

import operator
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING

from udfx.backends.diagnostics import CURRENT as DIAGNOSTICS
from udfx.backends.diagnostics import capture_lowering, record_lowering
from udfx.backends.factoring import factor_branches
from udfx.backends.graph import lower_block
from udfx.backends.graph_cost import graph_cost
from udfx.backends.hoisting import size
from udfx.backends.lowering_context import bindings_enabled, lowering_context, runtime_udfs
from udfx.backends.rules import Lowered, fold_predicate, knowing_parents
from udfx.backends.suffixes import dispatch_suffixes
from udfx.errors import Unsupported
from udfx.graphs import canonicalize, unique_size
from udfx.ir import FALSE, Call, Lit
from udfx.program import Program
from udfx.semantics.assumptions import assuming

if TYPE_CHECKING:
    from udfx.backends.base import Backend
    from udfx.backends.results import ResultLowering


EXPANDED_LIMIT = 10_000
GRAPH: ContextVar[bool | None] = ContextVar("udfx_graph_choice", default=None)


@contextmanager
def choosing_graph(graph: bool) -> Generator[None]:
    token = GRAPH.set(graph)
    try:
        yield
    finally:
        GRAPH.reset(token)


def build_smallest[E](backend: Backend[E], program: Program) -> E:
    expr = smallest(backend, program)
    if backend.size_limit is None or len(str(expr)) <= backend.size_limit:
        return expr
    return backend.compose(python_only(backend, program), program)


def python_only[E](backend: Backend[E], program: Program) -> Lowered[E]:
    details = DIAGNOSTICS.get()
    if details:
        details[:] = [replace(details[-1], exact=False)]
    return Lowered(backend.literal(Lit(None, program.result.or_none())), False)


def smallest[E](backend: Backend[E], program: Program) -> E:
    backend.check_names(program.schema)
    roots = canonicalize((program.value, program.fault))
    ordinary, shared = graph_cost(roots)
    chosen = GRAPH.get() is not None or not backend.choose_graphs or ordinary == shared
    if chosen or not backend.supports_graphs():
        return backend.build_candidate(program)
    outer = DIAGNOSTICS.get()
    candidates = []
    for graph in (False, True):
        with choosing_graph(graph), capture_lowering() as details:
            candidates.append((backend.build_candidate(program), details))
    native = [
        candidate
        for candidate in candidates
        if not any(detail.exact is False for detail in candidate[1])
    ]
    expr, details = min(
        native or candidates, key=lambda candidate: backend.expression_cost(candidate[0])
    )
    if outer is not None:
        outer[:] = details
    return expr


def lower_program[E](
    backend: Backend[E], program: Program, results: ResultLowering[E] | None = None
) -> Lowered[E]:
    if not program.deterministic:
        exact = backend.literal(FALSE)
        record_lowering(
            exact, value_guard=True, fault_guard=False, failure=False, native_errors=False
        )
        return Lowered(backend.literal(Lit(None, program.result.or_none())), exact)
    program = factored(backend, program)
    expanded = size(program.value) + size(program.fault)
    graph = False
    selected = program
    choice = GRAPH.get()
    if expanded > EXPANDED_LIMIT or bindings_enabled(backend.binds) or choice:
        value, fault = canonicalize((program.value, program.fault))
        if unique_size((value, fault)) <= EXPANDED_LIMIT:
            ordinary, shared = graph_cost((value, fault))
            preferred = shared < ordinary if choice is None else choice
            graph = (expanded > EXPANDED_LIMIT or preferred) and backend.supports_graphs()
        if graph:
            selected = replace(program, value=value, fault=fault)
    if expanded > EXPANDED_LIMIT and not graph:
        raise Unsupported(
            f"the expression has {expanded} nodes once shared values are expanded; "
            f"the limit is {EXPANDED_LIMIT}"
        )
    roots = (program.value, program.fault)
    try:
        with lowering_context(binds=False if graph else None, graph=graph), knowing_parents(roots):
            return lower_result(backend, selected, results, graph)
    except Unsupported:
        if not graph or expanded > EXPANDED_LIMIT:
            raise
    with lowering_context(), knowing_parents(roots):
        return lower_result(backend, program, results, False)


def factored[E](backend: Backend[E], program: Program) -> Program:
    def literal(call: Call, index: int) -> bool:
        return index in backend.rules.options(call.op.key).literals

    roots = canonicalize((program.value, program.fault))
    rewritten = factor_branches(dispatch_suffixes(roots, literal), literal)
    if all(map(operator.is_, rewritten, roots)):
        return program
    value, fault = canonicalize(rewritten)
    return replace(program, value=value, fault=fault)


def lower_result[E](
    backend: Backend[E], program: Program, results: ResultLowering[E] | None, graph: bool
) -> Lowered[E]:
    value_guard = fault_guard = failure_present = False

    def combine(values: Sequence[Lowered[E]]) -> Lowered[E]:
        nonlocal value_guard, fault_guard, failure_present
        value = values[0]
        fault = values[1] if len(values) == 2 else None
        value_guard = value.exact is not None
        fault_guard = fault is not None and (fault.exact is not None or fault.overflow is not None)
        failure_present = value.overflow is not None or fault is not None
        return value.map(
            lambda part: Lowered(
                part.expr,
                backend.conjunction(part.exact, backend.exact(fault))
                if fault is not None
                else part.exact,
                backend.disjunction(part.overflow, fault.expr)
                if fault is not None
                else part.overflow,
                guarantees=part.guarantees,
            )
        )

    predicate = fold_predicate(program.fault, backend.rules.rewrite_call)
    if graph:
        roots = (program.value,) if predicate == FALSE else (program.value, predicate)
        value = lower_block(backend, roots, {}, combine)
    else:
        with assuming(predicate, False):
            value = (
                backend.lower(program.value)
                if results is None
                else results.lower(program.value, {})
            )
        value = combine((value,) if predicate == FALSE else (value, backend.lower(predicate)))
    raised = backend.native_raise(program) if failure_present and value.exact is not False else None

    def finish(value: Lowered[E]) -> Lowered[E]:
        expr, exact, failure = value.expr, value.exact, value.overflow
        if exact is False:
            return Lowered(expr, False)
        if failure is not None:
            if raised is None:
                exact = backend.conjunction(exact, backend.negate(failure))
            else:
                condition = (
                    failure if exact is None else backend.both(failure, backend.predicate(exact))
                )
                expr = backend.select(condition, raised, expr)
        return Lowered(expr, exact, guarantees=value.guarantees)

    completed = value.map(finish)
    record_lowering(
        completed.exact,
        value_guard=value_guard,
        fault_guard=fault_guard,
        failure=failure_present or (results is not None and results.native_errors),
        native_errors=raised is not None or (results is not None and results.native_errors),
        runtime_udfs=runtime_udfs(),
    )
    return completed
