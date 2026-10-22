from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from udfx.frontend.core.logic import conjunction, disjunction, fold, negate
from udfx.ir import Call, Node
from udfx.ops.registry import OpKey
from udfx.types import Kind


@dataclass(frozen=True)
class Lowered[E]:
    expr: E
    exact: E | None = None
    overflow: E | None = None


type RuleResult[E] = E | Lowered[E]
type Rule[E] = Callable[..., RuleResult[E]]
type BinaryRule[E] = Callable[[E, E], RuleResult[E]]


BOOLEAN_RULES: dict[OpKey, Rule[Any]] = {
    OpKey("not", (Kind.BOOL,)): operator.invert,
    OpKey("and", (Kind.BOOL, Kind.BOOL)): operator.and_,
    OpKey("or", (Kind.BOOL, Kind.BOOL)): operator.or_,
}


def fold_predicate(node: Node) -> Node:
    if not isinstance(node, Call) or node.op.key not in BOOLEAN_RULES:
        return fold(node)
    arguments = tuple(fold_predicate(argument) for argument in node.arguments)
    builders = {"not": negate, "and": conjunction, "or": disjunction}
    return builders[node.op.name](*arguments)


@dataclass(frozen=True)
class RuleOptions:
    literals: tuple[int, ...] = ()
    facts: Mapping[str, Callable[[Call], object]] = field(default_factory=dict)
    guards: Mapping[int, Callable[[Any], Any]] = field(default_factory=dict)


class RuleSet[E]:
    def __init__(self) -> None:
        self._rules: dict[OpKey, Rule[E]] = dict(BOOLEAN_RULES)
        self._options: dict[OpKey, RuleOptions] = {}

    def __call__[**P](
        self,
        name: str,
        *kinds: Kind,
        literals: tuple[int, ...] = (),
        facts: Mapping[str, Callable[[Call], object]] | None = None,
        guards: Mapping[int, Callable[[Any], Any]] | None = None,
    ) -> Callable[[Callable[P, RuleResult[E]]], Callable[P, RuleResult[E]]]:
        key = OpKey(name, kinds)

        def decorate(rule: Callable[P, RuleResult[E]]) -> Callable[P, RuleResult[E]]:
            if key in self._rules:
                raise ValueError(f"duplicate rule {key}")
            self._rules[key] = rule
            self._options[key] = RuleOptions(literals, facts or {}, guards or {})
            return rule

        return decorate

    def get(self, key: OpKey) -> Rule[E] | None:
        return self._rules.get(key)

    def options(self, key: OpKey) -> RuleOptions:
        return self._options.get(key, RuleOptions())

    def keys(self) -> frozenset[OpKey]:
        return frozenset(self._rules)
