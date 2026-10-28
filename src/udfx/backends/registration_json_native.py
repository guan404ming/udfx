from functools import partial
from typing import Any

from udfx.backends.registration_json_items import register_json_items
from udfx.backends.rules import Lowered, RuleSet
from udfx.frontend.core.json import typed_get
from udfx.semantics.facts import (
    json_argument_validated,
    json_default_is_none,
    json_path_indexes,
    json_path_source,
)
from udfx.semantics.json import lower_decoded_strings, lower_valid, native_shape
from udfx.semantics.json_lowering import (
    JsonPrimitives,
    accepted,
    get,
    get_result,
    has_type,
    iterable,
    result,
    safe,
)
from udfx.semantics.json_values import SCALARS, result_name
from udfx.types import STR, Kind


def register_native_json(rules: RuleSet, primitives: JsonPrimitives) -> None:
    register_json_items(rules, primitives)
    rules("json.result.list[str]", Kind.JSON)(
        lambda value: Lowered(
            *lower_decoded_strings(
                primitives.text, value, primitives.strings, primitives.possessive
            )
        )
    )
    text = primitives.text
    facts = {"validated": json_argument_validated}

    def is_type(value: Any, name: Any, validated: bool = False) -> Lowered:
        if not isinstance(name, str):
            return Lowered(text.literal(False), text.literal(False))
        return Lowered(
            has_type(primitives, value, name), None if validated else accepted(primitives, value)
        )

    def valid(value: Any) -> Lowered:
        exact = text.when(
            native_shape(text, value, primitives.possessive),
            text.literal(True),
            accepted(primitives, value),
        )
        return Lowered(*lower_valid(text, value, exact))

    rules("json.valid", Kind.STR)(valid)
    rules("json.value", Kind.STR, facts=facts)(
        lambda value, validated=False: Lowered(
            safe(primitives, value), None if validated else accepted(primitives, value)
        )
    )
    for kind in (Kind.STR, Kind.JSON):
        rules("json.is_type", kind, Kind.STR, literals=(1,), facts=facts)(is_type)
    rules("json.none", Kind.JSON, facts=facts)(
        lambda value, validated=False: is_type(value, "NoneType", validated)
    )

    def lookup(
        value: Any,
        key: Any,
        default: Any,
        validated: bool = False,
        default_validated: bool = False,
        *,
        default_none: bool = False,
    ) -> Lowered:
        return Lowered(
            *get(
                primitives,
                value,
                key,
                default,
                validated=validated,
                default_validated=default_validated,
                default_none=default_none,
            )
        )

    def get_error(
        value: Any,
        key: Any,
        default: Any,
        validated: bool = False,
        *,
        source: Any = None,
        indexes: tuple[int, ...] = (),
    ) -> Lowered:
        return Lowered(
            ~has_type(primitives, value, "dict", source=source, indexes=indexes),
            None if validated else accepted(primitives, value),
        )

    get_facts = {**facts, "default_validated": partial(json_argument_validated, index=2)}
    for output in (STR, STR.or_none()):
        rules(
            f"json.get.{output}",
            Kind.JSON,
            Kind.STR,
            Kind.JSON,
            facts={
                **get_facts,
                "default_none": json_default_is_none,
                "indexes": json_path_indexes,
            },
            expressions={"source": json_path_source},
        )(
            lambda value, key, default, output=output, **facts: Lowered(
                *get_result(primitives, value, key, default, output, **facts)
            )
        )
        if primitives.typed_field is not None:
            rules.rewrite(result_name(output), Kind.JSON)(typed_get)
    for kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL):
        rules(
            "json.get",
            Kind.JSON,
            kind,
            Kind.JSON,
            facts={**get_facts, "default_none": json_default_is_none}
            if kind is Kind.STR
            else get_facts,
        )(
            lookup
            if kind is Kind.STR
            else lambda value, key, default, validated=False, default_validated=False: Lowered(
                safe(primitives, default, default_validated),
                accepted(primitives, default, default_validated),
            )
        )
        rules(
            "json.get.AttributeError",
            Kind.JSON,
            kind,
            Kind.JSON,
            facts={**facts, "indexes": json_path_indexes},
            expressions={"source": json_path_source} if primitives.object_at_path else {},
        )(get_error)
    rules("json.items", Kind.JSON, facts=facts)(
        lambda value, validated=False: Lowered(
            primitives.items(safe(primitives, value, validated)),
            accepted(primitives, value, validated) & has_type(primitives, value, "list"),
        )
    )
    rules("json.items.TypeError", Kind.JSON, facts=facts)(
        lambda value, validated=False: Lowered(
            ~iterable(primitives, value),
            accepted(primitives, value, validated),
        )
    )
    for type in SCALARS:
        for output in (type, type.or_none()):
            rules(result_name(output), Kind.JSON, facts=facts)(
                lambda value, output=output, validated=False: Lowered(
                    *result(primitives, value, output, validated)
                )
            )
