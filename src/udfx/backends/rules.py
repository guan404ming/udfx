from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from udfx.frontend.core.comprehensions import cardinality
from udfx.frontend.core.logic import conjunction, disjunction, fold, negate
from udfx.frontend.core.membership import comparison_membership, initial_lowercase_membership
from udfx.frontend.core.sequences import literal_string_item
from udfx.frontend.core.text import combined_search, joined_split, lowercase_contains
from udfx.ir import FALSE, TRUE, Call, Node
from udfx.ops.registry import OpKey
from udfx.semantics.assumptions import assuming
from udfx.semantics.ranges import proven
from udfx.types import Kind


@dataclass(frozen=True)
class Lowered[E]:
    expr: E
    exact: E | None = None
    overflow: E | None = None


type Scope[E] = Mapping[Node, E | Lowered[E]]


type RuleResult[E] = E | Lowered[E]
type Rule[E] = Callable[..., RuleResult[E]]
type BinaryRule[E] = Callable[[E, E], RuleResult[E]]
type Rewrite = Callable[[Call], Node | None]


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


@dataclass(frozen=True)
class RuleOptions:
    literals: tuple[int, ...] = ()
    facts: Mapping[str, Callable[[Call], object]] = field(default_factory=dict)
    guards: Mapping[int, Callable[[Any], Any]] = field(default_factory=dict)
    linear: bool = False
    expressions: Mapping[str, Callable[[Call], Node | None]] = field(default_factory=dict)


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
            OpKey("__contains__", (Kind.STR, Kind.STR)): lowercase_contains,
            OpKey("__contains__", (Kind.LIST, Kind.STR)): initial_lowercase_membership,
            OpKey("len", (Kind.LIST,)): cardinality,
            OpKey("bool", (Kind.LIST,)): cardinality,
            OpKey("__getitem__", (Kind.STR, Kind.INT)): literal_string_item,
        }

    def rewrite(self, name: str, *kinds: Kind) -> Callable[[Rewrite], Rewrite]:
        key = OpKey(name, kinds)

        def decorate(rewrite: Rewrite) -> Rewrite:
            if key in self._rewrites:
                raise ValueError(f"duplicate rewrite {key}")
            self._rewrites[key] = rewrite
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
    ) -> Callable[[Callable[P, RuleResult[E]]], Callable[P, RuleResult[E]]]:
        key = OpKey(name, kinds)

        def decorate(rule: Callable[P, RuleResult[E]]) -> Callable[P, RuleResult[E]]:
            if key in self._rules:
                raise ValueError(f"duplicate rule {key}")
            self._rules[key] = rule
            self._options[key] = RuleOptions(
                literals, facts or {}, guards or {}, linear, expressions or {}
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
