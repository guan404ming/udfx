from __future__ import annotations

import operator
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from udfx.backends.graph.siblings import share, shared
from udfx.backends.lowering.guarantees import Guarantee
from udfx.frontend.core.comprehensions import cardinality
from udfx.frontend.core.logic import conjunction, disjunction, fold, negate
from udfx.frontend.core.membership import comparison_membership, initial_lowercase_membership
from udfx.frontend.core.sequences import literal_string_item
from udfx.frontend.core.text import combined_search, joined_split, lowercase_contains, split_count
from udfx.graphs import parent_names
from udfx.ir import FALSE, TRUE, Call, Guarded, If, Node
from udfx.ops.registry import OpKey
from udfx.semantics.numbers.assumptions import assuming
from udfx.semantics.numbers.ranges import proven
from udfx.types import Kind


@dataclass(frozen=True)
class Lowered[E]:
    expr: E
    exact: E | Literal[False] | None = None
    overflow: E | None = None
    recover_overflow: Callable[[E], E] | None = None
    prepare: Callable[[Callable[[Lowered[E]], E]], E] | None = None
    guarantees: tuple[Guarantee, ...] = ()

    def consume(self, finish: Callable[[Lowered[E]], E]) -> E:
        return finish(self) if self.prepare is None else self.prepare(finish)

    def map(self, transform: Callable[[Lowered[E]], Lowered[E]], pure: bool = False) -> Lowered[E]:
        value = transform(replace(self, prepare=None))
        if self.prepare is not None:
            value = replace(
                value,
                prepare=lambda finish: self.consume(lambda part: transform(part).consume(finish)),
            )
        found = shared(self) if pure else None
        if found is None:
            return value
        base, view = found
        return share(value, base, lambda bound: transform(view(bound)))


type Scope[E] = Mapping[Node, E | Lowered[E]]


type RuleResult[E] = E | Lowered[E]
type Rule[E] = Callable[..., RuleResult[E]]
type BinaryRule[E] = Callable[[E, E], RuleResult[E]]
type Rewrite = Callable[[Call], Node | None]


PARENTS: ContextVar[Mapping[Node, frozenset[str]] | None] = ContextVar("udfx_parents", default=None)


@contextmanager
def knowing_parents(roots: tuple[Node, ...]) -> Generator[None]:
    token = PARENTS.set(parent_names(roots))
    try:
        yield
    finally:
        PARENTS.reset(token)


def parents(node: Node) -> frozenset[str]:
    try:
        return (PARENTS.get() or {}).get(node, frozenset())
    except TypeError:
        return frozenset()


def unshared_lowercase_contains(node: Call) -> Node | None:
    if parents(node.arguments[0]) - {"__contains__"}:
        return None
    return lowercase_contains(node)


def unshared_split_count(node: Call) -> Node | None:
    if parents(node.arguments[0]) != frozenset({"len"}):
        return None
    return split_count(node)


BOOLEAN_RULES: dict[OpKey, Rule[Any]] = {
    OpKey("not", (Kind.BOOL,)): operator.invert,
    OpKey("and", (Kind.BOOL, Kind.BOOL)): operator.and_,
    OpKey("or", (Kind.BOOL, Kind.BOOL)): operator.or_,
}


def fold_predicate(node: Node, rewrite: Rewrite) -> Node:
    if isinstance(node, Call):
        rewritten = rewrite(node)
        if rewritten is not None and rewritten != node:
            return fold_predicate(rewritten, rewrite)
    known = proven(node)
    if known is not None:
        return TRUE if known else FALSE
    if not isinstance(node, Call) or node.op.key not in BOOLEAN_RULES:
        return fold(node)
    first, *rest = node.arguments
    arguments = (fold_predicate(first, rewrite),)
    if rest:
        with assuming(arguments[0], node.op.name == "and"):
            arguments += (fold_predicate(rest[0], rewrite),)
    builders = {"not": negate, "and": conjunction, "or": disjunction}
    return builders[node.op.name](*arguments)


def fold_value(node: Node, rewrite: Rewrite) -> Node:
    while isinstance(node, If | Guarded):
        predicate = node.condition if isinstance(node, If) else node.exact
        known = fold_predicate(predicate, rewrite)
        if isinstance(node, If) and known in (TRUE, FALSE):
            node = node.then if known == TRUE else node.orelse
        elif isinstance(node, Guarded) and known == TRUE:
            node = node.value
        else:
            break
    return node


@dataclass(frozen=True)
class RuleOptions:
    literals: tuple[int, ...] = ()
    facts: Mapping[str, Callable[[Call], object]] = field(default_factory=dict)
    guards: Mapping[int, Callable[[Any], Any]] = field(default_factory=dict)
    linear: bool = False
    expressions: Mapping[str, Callable[[Call], Node | None]] = field(default_factory=dict)
    proves: Callable[[Call], Node | None] | None = None
    false_exact: bool = False


class RuleSet[E]:
    def __init__(self) -> None:
        self._rules: dict[OpKey, Rule[E]] = dict(BOOLEAN_RULES)
        self._options: dict[OpKey, RuleOptions] = {}
        self._rewrites: dict[OpKey, Rewrite] = {
            OpKey("or", (Kind.BOOL, Kind.BOOL)): (
                lambda node: comparison_membership(node) or combined_search(node)
            ),
            OpKey("and", (Kind.BOOL, Kind.BOOL)): comparison_membership,
            OpKey("__add__", (Kind.STR, Kind.STR)): joined_split,
            OpKey("__contains__", (Kind.STR, Kind.STR)): unshared_lowercase_contains,
            OpKey("__contains__", (Kind.LIST, Kind.STR)): initial_lowercase_membership,
            OpKey("len", (Kind.LIST,)): lambda node: (
                unshared_split_count(node) or cardinality(node)
            ),
            OpKey("bool", (Kind.LIST,)): cardinality,
            OpKey("__getitem__", (Kind.STR, Kind.INT)): literal_string_item,
        }
        self._registered: set[OpKey] = set()

    def rewrite(self, name: str, *kinds: Kind) -> Callable[[Rewrite], Rewrite]:
        key = OpKey(name, kinds)

        def decorate(rewrite: Rewrite) -> Rewrite:
            if key in self._registered:
                raise ValueError(f"duplicate rewrite {key}")
            self._registered.add(key)
            shared = self._rewrites.get(key)
            self._rewrites[key] = (
                rewrite if shared is None else lambda node: shared(node) or rewrite(node)
            )
            return rewrite

        return decorate

    def rewrite_call(self, node: Call) -> Node | None:
        rewrite = self._rewrites.get(node.op.key)
        return None if rewrite is None else rewrite(node)

    def __call__[**P](
        self,
        name: str,
        *kinds: Kind,
        literals: tuple[int, ...] = (),
        facts: Mapping[str, Callable[[Call], object]] | None = None,
        guards: Mapping[int, Callable[[Any], Any]] | None = None,
        linear: bool = False,
        expressions: Mapping[str, Callable[[Call], Node | None]] | None = None,
        proves: Callable[[Call], Node | None] | None = None,
        false_exact: bool = False,
    ) -> Callable[[Callable[P, RuleResult[E]]], Callable[P, RuleResult[E]]]:
        key = OpKey(name, kinds)

        def decorate(rule: Callable[P, RuleResult[E]]) -> Callable[P, RuleResult[E]]:
            if key in self._rules:
                raise ValueError(f"duplicate rule {key}")
            self._rules[key] = rule
            self._options[key] = RuleOptions(
                literals, facts or {}, guards or {}, linear, expressions or {}, proves, false_exact
            )
            return rule

        return decorate

    def get(self, key: OpKey) -> Rule[E] | None:
        return self._rules.get(key)

    def options(self, key: OpKey) -> RuleOptions:
        return self._options.get(key, RuleOptions())

    def linear(self, node: Call) -> bool:
        return self.options(node.op.key).linear

    def keys(self) -> frozenset[OpKey]:
        return frozenset(self._rules)
