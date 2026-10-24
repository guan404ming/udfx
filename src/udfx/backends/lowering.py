from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

from udfx.backends.diagnostics import record_lowering
from udfx.backends.hoisting import exact_free, hoist, invariant, size
from udfx.backends.indexing import (
    Parts,
    comprehension_parts,
    comprehension_result,
    indexed_source,
    literal_range,
    lower_indexed,
)
from udfx.backends.lowering_context import bindings_enabled
from udfx.backends.reductions import lower_reduction
from udfx.backends.registration import Lowered, RuleResult
from udfx.backends.rules import fold_predicate
from udfx.errors import Unsupported
from udfx.ir import FALSE, TRUE, Call, Comprehension, If, Lit, Node
from udfx.ops import OpKey
from udfx.program import Program
from udfx.semantics.assumptions import assuming
from udfx.types import Kind

if TYPE_CHECKING:
    from udfx.backends.base import Backend


EXPANDED_LIMIT = 10_000


def lower_program[E](backend: Backend[E], program: Program) -> Lowered[E]:
    expanded = size(program.value) + size(program.fault)
    if expanded > EXPANDED_LIMIT:
        raise Unsupported(
            f"the expression has {expanded} nodes once shared values are expanded; "
            f"the limit is {EXPANDED_LIMIT}"
        )
    predicate = fold_predicate(program.fault, backend.rules.rewrite_call)
    with assuming(predicate, False):
        value = backend.lower(program.value)
    expr, exact, failure = value.expr, value.exact, value.overflow
    fault_guard = False
    native_errors = False
    if predicate != FALSE:
        fault = backend.lower(predicate)
        fault_guard = fault.exact is not None or fault.overflow is not None
        exact = backend.conjunction(exact, backend.exact(fault))
        failure = backend.disjunction(failure, fault.expr)
    if failure is not None:
        raised = backend.native_raise(program)
        native_errors = raised is not None
        if raised is None:
            exact = backend.conjunction(exact, backend.negate(failure))
        else:
            condition = failure if exact is None else backend.both(failure, exact)
            expr = backend.select(condition, raised, expr)
    record_lowering(
        exact,
        value_guard=value.exact is not None,
        fault_guard=fault_guard,
        failure=failure is not None,
        native_errors=native_errors,
    )
    return Lowered(expr, exact)


def lower_if[E](backend: Backend[E], node: If, elements: Mapping[Node, E]) -> Lowered[E]:
    predicate = fold_predicate(node.condition, backend.rules.rewrite_call)
    if predicate in (TRUE, FALSE):
        return backend.lower(node.then if predicate == TRUE else node.orelse, elements)
    condition = backend.lower(predicate, elements)
    with assuming(predicate, True):
        then = backend.lower(node.then, elements)
    with assuming(predicate, False):
        orelse = backend.lower(node.orelse, elements)
    expr = backend.select(condition.expr, then.expr, orelse.expr)
    exact = backend.exact(condition)
    if then.exact is not None or orelse.exact is not None:
        branch = backend.select(
            condition.expr,
            then.exact if then.exact is not None else backend.literal(TRUE),
            orelse.exact if orelse.exact is not None else backend.literal(TRUE),
        )
        exact = backend.conjunction(exact, branch)
    overflow = None
    if then.overflow is not None or orelse.overflow is not None:
        overflow = backend.select(
            condition.expr,
            then.overflow if then.overflow is not None else backend.literal(FALSE),
            orelse.overflow if orelse.overflow is not None else backend.literal(FALSE),
        )
    return Lowered(expr, exact, overflow)


def lower_call[E](backend: Backend[E], node: Call, elements: Mapping[Node, E]) -> Lowered[E]:
    reduced = lower_reduction(backend, node, elements)
    if reduced is not None:
        return reduced
    if node.type.kind is Kind.BOOL:
        folded = fold_predicate(node, backend.rules.rewrite_call)
        if folded != node:
            return backend.lower(folded, elements)
    rewritten = backend.rules.rewrite_call(node)
    if rewritten is not None and rewritten != node:
        return backend.lower(rewritten, elements)
    rule = backend.rules.get(node.op.key)
    if rule is None:
        raise Unsupported(f"{backend.name} has no rule for {node.op.key}")
    options = backend.rules.options(node.op.key)
    arguments = [
        Lowered(cast("Any", argument.value))
        if index in options.literals and isinstance(argument, Lit)
        else backend.lower(argument, elements)
        for index, argument in enumerate(node.arguments)
    ]
    result: RuleResult[E] = rule(
        *(argument.expr for argument in arguments),
        **{name: fact(node) for name, fact in options.facts.items()},
    )
    lowered = cast("Lowered[E]", result) if isinstance(result, Lowered) else Lowered(result)
    guards = [
        guard(argument.value if isinstance(argument, Lit) else arguments[index].expr)
        for index, guard in options.guards.items()
        for argument in (node.arguments[index],)
    ]
    return Lowered(
        lowered.expr,
        backend.conjunction(
            *(backend.exact(argument) for argument in arguments), lowered.exact, *guards
        ),
        lowered.overflow,
    )


def lower_comprehension[E](
    backend: Backend[E], node: Comprehension, elements: Mapping[Node, E]
) -> Lowered[E]:
    hoisted = invariant(node, elements) if bindings_enabled(backend.binds) else None
    if hoisted is not None and exact_free(backend, hoisted, elements):
        with suppress(Unsupported):
            return hoist(backend, node, elements, hoisted)
    if backend.indexes and (rewritten := rewrite_range(backend, node, elements)) is not None:
        return rewritten
    source = backend.lower(node.source, elements)

    guarded = False

    def parts(element: E) -> Parts[E]:
        nonlocal guarded
        found = comprehension_parts(backend, node, {**elements, node.element: element})
        guarded = guarded or found[2] is not None
        return found

    try:
        if isinstance(node.source, Lit) and isinstance(node.source.value, tuple | list):
            chunks: list[E] = []
            exacts: list[E | None] = []
            faults: list[E | None] = []
            for item in node.source.value:
                element = backend.lower(Lit(item, node.element.type), elements)
                condition, value, exact, fault = parts(element.expr)
                chunks.append(
                    backend.select(
                        condition.expr,
                        backend.list_value([value.expr], node.type),
                        backend.literal(Lit((), node.type)),
                    )
                )
                exacts.append(exact)
                faults.append(fault if node.fault != FALSE else None)
            result = backend.literal(Lit((), node.type))
            concatenate = backend.rules.get(OpKey("__add__", (Kind.LIST, Kind.LIST)))
            if concatenate is None:
                raise Unsupported(f"{backend.name} has no list concatenation")
            for chunk in chunks:
                result = cast("E", concatenate(result, chunk))
            return Lowered(
                result,
                backend.conjunction(backend.exact(source), *exacts),
                backend.disjunction(*faults),
            )
        filtered = source.expr
        if node.condition != TRUE:
            filtered = backend.list_filter(filtered, lambda element: parts(element)[0].expr)
        value = filtered
        if node.value != node.element:
            value = backend.list_transform(filtered, lambda element: parts(element)[1].expr)
        return comprehension_result(
            backend,
            node,
            source,
            value,
            guarded=guarded,
            flags=lambda pick: backend.list_transform(
                source.expr, lambda element: pick(parts(element))
            ),
        )
    except Unsupported:
        if node.requires_native:
            raise
    rewritten = rewrite_range(backend, node, elements)
    if rewritten is not None:
        return rewritten
    return Lowered(backend.literal(Lit((), node.type)), backend.literal(FALSE))


def rewrite_range[E](
    backend: Backend[E], node: Comprehension, elements: Mapping[Node, E]
) -> Lowered[E] | None:
    with suppress(Unsupported):
        unrolled = literal_range(node)
        if unrolled is not None:
            return lower_comprehension(backend, unrolled, elements)
        indexed = indexed_source(node)
        if indexed is not None:
            return lower_indexed(backend, node, elements, *indexed)
    return None
