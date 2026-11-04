from typing import Any

import pytest

from udfx.backends.engines.duckdb import DuckDBBackend
from udfx.backends.graph import bindings as graph
from udfx.backends.graph import cost as graph_cost
from udfx.backends.lowering import programs as program_lowering
from udfx.backends.lowering.context import lowering_context
from udfx.backends.rewrites.hoisting import worth_binding
from udfx.backends.rules import Lowered
from udfx.errors import Unsupported
from udfx.ir import FALSE, TRUE, Col, Comprehension, Element, Node, children
from udfx.ops import apply
from udfx.program import Program
from udfx.types import STR, list_of


def repeated(depth: int) -> Node:
    value: Node = Col("text", STR)
    for _ in range(depth):
        value = apply(".lower", value)
    return apply("__add__", value, value)


@pytest.mark.parametrize("depth", [0, 1, 12])
def test_operation_count_matches_expanded_tree(depth: int) -> None:
    shared = repeated(depth)
    left = right = Col("text", STR)
    for _ in range(depth):
        left, right = apply(".lower", left), apply(".lower", right)
    tree = apply("__add__", left, right)
    assert graph_cost.operation_count(shared) == 2 * depth + 1
    assert graph_cost.operation_count(tree) == graph_cost.operation_count(shared)


def test_operation_count_visits_shared_nodes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    value: Node = Col("text", STR)
    for _ in range(40):
        value = apply("__add__", value, value)
    visits = 0

    def counted(node: Node) -> tuple[Node, ...]:
        nonlocal visits
        visits += 1
        return children(node)

    monkeypatch.setattr(graph_cost, "children", counted)
    assert graph_cost.operation_count(value) == 2**40 - 1
    assert visits == 41
    assert worth_binding(value, 2)


def test_comprehension_dependencies_visit_shared_nodes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = Element("item", STR)
    value: Node = element
    for _ in range(20):
        value = apply("__add__", value, value)
    comprehension = Comprehension(Col("items", list_of(STR)), element, value, TRUE, FALSE)
    visits = 0

    def counted(node: Node) -> tuple[Node, ...]:
        nonlocal visits
        visits += 1
        return children(node)

    monkeypatch.setattr(graph, "children", counted)
    graph.layers([comprehension], {})
    assert visits <= 22


@pytest.mark.parametrize("depth, expected", [(10, False), (11, False), (12, True)])
def test_graph_requires_net_operation_savings(depth: int, expected: bool) -> None:
    ordinary, shared = graph_cost.graph_cost((repeated(depth),))
    assert (shared < ordinary) is expected


@pytest.mark.parametrize(
    "default, override, expected",
    [(True, None, True), (False, None, False), (True, False, False), (False, True, True)],
)
def test_early_graph_honors_binding_scope(
    default: bool, override: bool | None, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = DuckDBBackend()
    monkeypatch.setattr(backend, "binds", default)
    monkeypatch.setattr(backend, "supports_graphs", lambda: True)
    selected: list[bool] = []

    def finish(*arguments: Any) -> Lowered[Any]:
        selected.append(arguments[-1])
        return Lowered(None)

    monkeypatch.setattr(program_lowering, "lower_result", finish)
    program = Program(lambda text: text.lower() * 2, {"text": STR}, repeated(12), FALSE)
    with lowering_context(binds=override):
        program_lowering.lower_program(backend, program)
    assert selected == [expected]


@pytest.mark.parametrize("large, supported", [(False, False), (True, True), (True, False)])
def test_graph_limit_rescue_is_independent_of_binding_preference(
    large: bool, supported: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = DuckDBBackend()
    monkeypatch.setattr(backend, "binds", False)
    monkeypatch.setattr(backend, "supports_graphs", lambda: supported)
    monkeypatch.setattr(program_lowering, "EXPANDED_LIMIT", 20 if large else 10_000)
    selected: list[bool] = []

    def finish(*arguments: Any) -> Lowered[Any]:
        selected.append(arguments[-1])
        return Lowered(None)

    monkeypatch.setattr(program_lowering, "lower_result", finish)
    program = Program(lambda text: text.lower() * 2, {"text": STR}, repeated(12), FALSE)
    if large and not supported:
        with pytest.raises(Unsupported, match="once shared values are expanded"):
            program_lowering.lower_program(backend, program)
        assert selected == []
    else:
        program_lowering.lower_program(backend, program)
        assert selected == [large]


def test_cheap_expression_does_not_probe_graph_support(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = DuckDBBackend()

    def unexpected() -> bool:
        pytest.fail("Graph support was checked without operation savings")

    monkeypatch.setattr(backend, "supports_graphs", unexpected)
    program = Program(lambda text: text.lower() * 2, {"text": STR}, repeated(1), FALSE)
    assert backend.lower_program(program).prepare is None


@pytest.mark.parametrize("required", [False, True])
def test_optional_graph_failure_retries_original_native_lowering(
    required: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = DuckDBBackend()
    monkeypatch.setattr(backend, "supports_graphs", lambda: True)
    monkeypatch.setattr(program_lowering, "EXPANDED_LIMIT", 20 if required else 10_000)
    selected: list[bool] = []
    program = Program(lambda text: text.lower() * 2, {"text": STR}, repeated(12), FALSE)

    def finish(*arguments: Any) -> Lowered[Any]:
        graph = arguments[-1]
        selected.append(graph)
        if graph:
            raise Unsupported("Graph binding is not supported")
        assert arguments[1] is program
        return Lowered(None)

    monkeypatch.setattr(program_lowering, "lower_result", finish)
    if required:
        with pytest.raises(Unsupported, match="Graph binding is not supported"):
            program_lowering.lower_program(backend, program)
        assert selected == [True]
    else:
        program_lowering.lower_program(backend, program)
        assert selected == [True, False]
