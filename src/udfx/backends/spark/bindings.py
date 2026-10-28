from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from pyspark.sql import Column, functions

from udfx.backends.indexing import matching
from udfx.backends.lowering_context import lowering_context
from udfx.backends.rules import Scope
from udfx.ir import Comprehension, Node
from udfx.types import Kind


@dataclass
class BindingScope:
    requires_unbound: bool = False
    depth: int = 0
    numeric: bool = True


CURRENT: ContextVar[BindingScope | None] = ContextVar("udfx spark bindings", default=None)


@contextmanager
def binding_scope() -> Generator[BindingScope]:
    scope = BindingScope()
    token = CURRENT.set(scope)
    try:
        yield scope
    finally:
        CURRENT.reset(token)


@contextmanager
def reading(node: Node, elements: Scope[Column] | None) -> Generator[None]:
    scope = CURRENT.get()
    if scope is None:
        yield
        return
    numeric = node.type.kind in {Kind.BOOL, Kind.INT, Kind.FLOAT}
    if elements and node in elements and (not scope.numeric or not numeric):
        scope.requires_unbound = True
    if (
        isinstance(node, Comprehension)
        and elements
        and matching(node, lambda part: part in elements)
    ):
        scope.requires_unbound = True
    previous, scope.numeric = scope.numeric, numeric
    try:
        yield
    finally:
        scope.numeric = previous


def bind(value: Column, body: Callable[[Column], Column], *, tracked: bool = False) -> Column:
    scope = CURRENT.get()
    if scope is not None:
        scope.requires_unbound |= not tracked or scope.depth > 0
        scope.depth += 1
    try:
        expr = functions.transform(functions.array(value), body)[0]
    finally:
        if scope is not None:
            scope.depth -= 1
    with lowering_context() as context:
        context.has_bindings = True
    return expr
