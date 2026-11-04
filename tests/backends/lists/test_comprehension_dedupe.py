from typing import Any, cast

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.engines.duckdb import DuckDBBackend
from udfx.backends.graph import bindings as graph
from udfx.backends.graph.bindings import layers
from udfx.backends.graph.siblings import siblings
from udfx.backends.lowering.programs import choosing_graph
from udfx.frontend import trace
from udfx.graphs import canonicalize
from udfx.ir import Col, Comprehension, If, Node, children, lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import INT, STR

GRAPH_BACKENDS = [
    value for value in backend_params() if cast("Backend[Any]", value.values[0]).supports_graphs()
]
SUFFIXES = ("ational", "tional", "enci", "anci", "izer", "abli", "alli", "entli", "eli", "ousli")
ROWS = [("",), ("b a b",), ("rational rational sizer",), ("abcd abcd éabc",), ("12345 é",)]


def rewrite(word: str) -> str:
    for suffix in SUFFIXES:
        if word.endswith(suffix):
            return word[: len(word) - len(suffix)] + suffix.upper()
    return word.lower() + word[3]


def rewritten(text: str) -> list[str]:
    try:
        return [rewrite(word) for word in text.split()]
    except IndexError:
        return ["?"]


def doubled(text: str) -> list[int]:
    try:
        return [int(word) * 2 for word in text.split()]
    except ValueError:
        return [-1]


def comprehensions(roots: tuple[Node, ...]) -> list[Comprehension]:
    found: dict[int, Comprehension] = {}
    pending = list(roots)
    while pending:
        node = pending.pop()
        if isinstance(node, Comprehension):
            found[id(node)] = node
        pending.extend(children(node))
    return list(found.values())


@pytest.mark.parametrize("backend", GRAPH_BACKENDS)
def test_repeated_elements_lower_once(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    distinct = backend.list_distinct

    def counted(source: Any, transform: Any) -> Any:
        calls.append(source)
        return distinct(source, transform)

    monkeypatch.setattr(backend, "list_distinct", counted)
    with choosing_graph(True):
        assert check_function(rewritten, {"text": STR}, backend, ROWS) == []
    assert calls


def test_fault_comprehension_shares_its_source() -> None:
    program = trace(doubled, {"text": STR})
    found = comprehensions(canonicalize((program.value, program.fault)))
    assert len(found) == 2
    assert siblings(found) == {id(node): tuple(found) for node in found}


@pytest.mark.parametrize("backend", GRAPH_BACKENDS)
def test_shared_comprehensions_match_python(backend: Backend[Any]) -> None:
    rows = [("",), ("1 2 1",), ("1 x",), ("٣ 4",)]
    with choosing_graph(True):
        assert check_function(doubled, {"text": STR}, backend, rows) == []


def test_shared_comprehension_packs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    def size() -> int:
        with choosing_graph(True):
            return len(str(DuckDBBackend().build(trace(doubled, {"text": STR}))))

    packed = size()
    monkeypatch.setattr(graph, "shared", lambda value: None)
    assert packed < size()


def test_cheap_shared_integers_are_recomputed() -> None:
    text = Col("text", STR)
    size = apply("len", text)
    found = apply("re.contains", text, lit("a+"), lit(0))
    root = apply(
        "__add__",
        apply("__mul__", size, size),
        apply("__add__", If(found, lit(1), lit(2), INT), If(found, lit(3), size, INT)),
    )
    _, groups = layers((root,), {})
    grouped = [node for group in groups for node in group]
    assert all(node is not size for node in grouped)
    assert any(node is found for node in grouped)
