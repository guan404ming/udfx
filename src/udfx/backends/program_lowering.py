from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from udfx.backends.diagnostics import record_lowering
from udfx.backends.graph import lower_graph
from udfx.backends.hoisting import size
from udfx.backends.lowering_context import lowering_context, runtime_udfs
from udfx.backends.rules import Lowered, fold_predicate
from udfx.errors import Unsupported
from udfx.graphs import canonicalize, unique_size
from udfx.ir import FALSE, Lit, Node
from udfx.program import Program
from udfx.semantics.assumptions import assuming

if TYPE_CHECKING:
    from udfx.backends.base import Backend
    from udfx.backends.results import ResultLowering


EXPANDED_LIMIT = 10_000


def lower_program[E](
    backend: Backend[E], program: Program, results: ResultLowering[E] | None = None
) -> Lowered[E]:
    if not program.deterministic:
        exact = backend.literal(FALSE)
        record_lowering(
            exact, value_guard=True, fault_guard=False, failure=False, native_errors=False
        )
        return Lowered(backend.literal(Lit(None, program.result.or_none())), exact)
    expanded = size(program.value) + size(program.fault)
    graph = expanded > EXPANDED_LIMIT and backend.supports_graphs()
    if graph:
        value, fault = canonicalize((program.value, program.fault))
        program = replace(program, value=value, fault=fault)
        graph = unique_size((value, fault)) <= EXPANDED_LIMIT
    if expanded > EXPANDED_LIMIT and not graph:
        raise Unsupported(
            f"the expression has {expanded} nodes once shared values are expanded; "
            f"the limit is {EXPANDED_LIMIT}"
        )
    with lowering_context(binds=False if graph else None, graph=graph):
        return lower_result(backend, program, results, graph)


def lower_result[E](
    backend: Backend[E], program: Program, results: ResultLowering[E] | None, graph: bool
) -> Lowered[E]:
    def lower(node: Node) -> Lowered[E]:
        return lower_graph(backend, node, {}) if graph else backend.lower(node)

    predicate = (
        program.fault if graph else fold_predicate(program.fault, backend.rules.rewrite_call)
    )
    with assuming(predicate, False):
        value = (
            lower(program.value) if graph or results is None else results.lower(program.value, {})
        )
    fault = None if predicate == FALSE else lower(predicate)
    fault_guard = fault is not None and (fault.exact is not None or fault.overflow is not None)
    failure = value.overflow is not None or fault is not None
    raised = backend.native_raise(program) if failure else None

    def finish(value: Lowered[E]) -> Lowered[E]:
        expr, exact, failure = value.expr, value.exact, value.overflow
        if fault is not None:
            exact = backend.conjunction(exact, backend.exact(fault))
            failure = backend.disjunction(failure, fault.expr)
        if failure is not None:
            if raised is None:
                exact = backend.conjunction(exact, backend.negate(failure))
            else:
                condition = failure if exact is None else backend.both(failure, exact)
                expr = backend.select(condition, raised, expr)
        return Lowered(expr, exact, guarantees=value.guarantees)

    completed = value.map(finish)
    record_lowering(
        completed.exact,
        value_guard=value.exact is not None,
        fault_guard=fault_guard,
        failure=failure or (results is not None and results.native_errors),
        native_errors=raised is not None or (results is not None and results.native_errors),
        runtime_udfs=runtime_udfs(),
    )
    return completed
