from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from udfx.ir import Element, Node


@dataclass
class Bindings:
    values: Mapping[Node, Node]
    ranges: dict[tuple[Node, int], tuple[int, int] | None] = field(default_factory=dict)


SOURCES: ContextVar[Bindings | None] = ContextVar("udfx_bound_sources", default=None)


def source(node: Node) -> Node:
    current = SOURCES.get()
    sources = {} if current is None else current.values
    while isinstance(node, Element) and node in sources:
        node = sources[node]
    return node


@contextmanager
def bound_sources(values: Mapping[Node, Node]) -> Generator[None]:
    current = SOURCES.get()
    token = SOURCES.set(Bindings({**({} if current is None else current.values), **values}))
    try:
        yield
    finally:
        SOURCES.reset(token)
