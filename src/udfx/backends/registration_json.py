from collections.abc import Callable

from udfx.backends.lowering_context import EXACT
from udfx.backends.registration_json_native import register_native_json
from udfx.backends.rules import Lowered, RuleSet
from udfx.frontend.core.json import decoded_string, decoded_strings, known_type, serialized_strings
from udfx.frontend.core.json import text as json_text
from udfx.ir import FALSE, TRUE, Call, Comprehension, Element, Guarded, Lit, Node, lit
from udfx.ops import OpKey, apply
from udfx.ops.json import OPERATIONS
from udfx.semantics.bindings import source
from udfx.semantics.facts import json_argument_validated, json_string_is_native
from udfx.semantics.json import STRING, lower_decoded_string, lower_type, lower_valid, native_shape
from udfx.semantics.json_lowering import JsonPrimitives
from udfx.semantics.literals import TextPrimitives
from udfx.semantics.ranges import proven
from udfx.types import STR, Kind


def register_json_string_decoder[E](
    rules: RuleSet[E], text: TextPrimitives, decode: Callable[[E], E]
) -> None:
    for name, nullable in (("json.result.str", False), ("json.result.str?", True)):
        rules(name, Kind.JSON, facts={"validated": json_string_is_native})(
            lambda value, nullable=nullable, validated=False: Lowered(
                *lower_decoded_string(text, value, decode, nullable, validated)
            )
        )


def register_json_values[E](
    rules: RuleSet[E],
    literal: Callable[[Lit], E],
    text: TextPrimitives,
    native: JsonPrimitives | None = None,
    encode_strings: Callable[[E, bool], E] | None = None,
    *,
    serialize_strings: Callable[[Node, Node], Node] = serialized_strings,
) -> None:
    rules.rewrite("json.is_type", Kind.STR, Kind.STR)(known_type)

    def serialize(node: Call) -> Node | None:
        values, ensure_ascii = node.arguments
        if (
            encode_strings is not None
            and values.type.item is not None
            and values.type.item.kind is Kind.STR
            and isinstance(ensure_ascii, Lit)
            and isinstance(ensure_ascii.value, bool)
        ):
            return None
        return serialize_strings(values, ensure_ascii)

    rules.rewrite("json.dumps", Kind.LIST, Kind.BOOL)(serialize)
    if encode_strings is not None:
        rules("json.dumps", Kind.LIST, Kind.BOOL, literals=(1,))(encode_strings)
    rules.rewrite("json.encode", Kind.STR)(
        lambda node: apply("json.dumps", node.arguments[0], FALSE)
    )
    if native is None:
        register_string_json(rules, text)
    else:
        rules.rewrite("json.is_type", Kind.JSON, Kind.STR)(known_type)
        register_native_json(rules, native)
    for op in {op.key: op for op in OPERATIONS}.values():
        if rules.get(op.key) is not None:
            continue
        if op.name in {"json.value", "json.text"}:
            rules(op.name, *op.key.kinds)(lambda value: value)
        else:
            rules(op.name, *op.key.kinds)(
                lambda *arguments, result=op.result: Lowered(
                    literal(Lit(None, result.or_none())), literal(FALSE)
                )
            )


def register_string_json[E](rules: RuleSet[E], text: TextPrimitives) -> None:
    def tokens(node: Call) -> Node:
        return apply("json.tokens", json_text(node.arguments[0]))

    def items(node: Call) -> Node:
        element = Element("json item", STR)
        return Comprehension(tokens(node), element, apply("json.value", element), TRUE, FALSE)

    rules("json.valid", Kind.STR)(
        lambda value: Lowered(*lower_valid(text, value, native_shape(text, value, False)))
    )
    for name in ("json.array_native", "json.string_native"):
        rules.rewrite(name, Kind.STR)(validated_shape)
    rules(
        "json.is_type",
        Kind.STR,
        Kind.STR,
        literals=(1,),
        facts={"validated": json_argument_validated},
    )(
        lambda value, name, validated=False: Lowered(
            *lower_type(text, value, name, False, validated)
        )
    )
    rules.rewrite("json.items", Kind.JSON)(items)
    rules.rewrite("json.length", Kind.JSON)(
        lambda node: Guarded(
            apply("re.count", json_text(node.arguments[0]), lit(STRING), lit(0)),
            apply("json.array_native", json_text(node.arguments[0])),
        )
    )
    rules.rewrite("json.item", Kind.JSON, Kind.INT)(
        lambda node: apply("json.value", apply("__getitem__", tokens(node), node.arguments[1]))
    )
    rules.rewrite("json.is_type", Kind.JSON, Kind.STR)(
        lambda node: (
            known_type(node)
            or apply("json.is_type", json_text(node.arguments[0]), node.arguments[1])
        )
    )
    for name in ("json.result.str", "json.result.str?"):
        if rules.get(OpKey(name, (Kind.JSON,))) is None:
            rules.rewrite(name, Kind.JSON)(
                lambda node: decoded_string(json_text(node.arguments[0]))
            )
    name = "json.result.list[str]"
    if rules.get(OpKey(name, (Kind.JSON,))) is None:
        rules.rewrite(name, Kind.JSON)(lambda node: decoded_strings(json_text(node.arguments[0])))


def validated_shape(node: Call) -> Node | None:
    value = source(node.arguments[0])
    for predicate in (EXACT.get() or {}).values():
        if (
            isinstance(predicate, Call)
            and predicate.op.name == "json.valid"
            and source(predicate.arguments[0]) == value
            and proven(predicate) is True
        ):
            kind = "list" if node.op.name == "json.array_native" else "str"
            return apply("json.is_type", node.arguments[0], lit(kind))
    return None
