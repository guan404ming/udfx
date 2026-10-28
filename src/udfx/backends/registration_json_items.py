from typing import Any

from udfx.backends.rules import Lowered, RuleSet
from udfx.semantics.facts import json_argument_type, json_argument_validated
from udfx.semantics.json_lowering import JsonPrimitives, accepted, has_type, item, iterable, safe
from udfx.types import Kind


def register_json_items(rules: RuleSet, primitives: JsonPrimitives) -> None:
    facts = {"validated": json_argument_validated}

    for kind in (Kind.STR, Kind.INT):
        for name, offset in (
            ("json.item", 0),
            ("json.item.TypeError", 2),
            ("json.item.KeyError", 3),
            ("json.item.IndexError", 4),
        ):

            def rule(
                value: Any,
                key: Any,
                *,
                kind: Kind = kind,
                offset: int = offset,
                validated: bool = False,
                container: str | None = None,
            ) -> Lowered:
                parts = item(
                    primitives,
                    value,
                    key,
                    integer=kind is Kind.INT,
                    validated=validated,
                    container=container,
                )
                return Lowered(parts[offset], parts[1])

            rules(name, Kind.JSON, kind, facts={**facts, "container": json_argument_type})(rule)
    rules("json.length", Kind.JSON, facts={**facts, "container": json_argument_type})(
        lambda value, validated=False, container=None: Lowered(
            primitives.size(safe(primitives, value, validated)),
            None
            if validated and container == "list"
            else accepted(primitives, value, validated) & has_type(primitives, value, "list"),
        )
    )
    rules("json.length.TypeError", Kind.JSON, facts=facts)(
        lambda value, validated=False: Lowered(
            ~iterable(primitives, value),
            accepted(primitives, value, validated),
        )
    )
