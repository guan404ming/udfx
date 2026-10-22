from __future__ import annotations

from collections.abc import Callable
from typing import Any

from udfx.backends.rules import Lowered, Rule, RuleSet
from udfx.semantics.facts import argument_length
from udfx.semantics.literals import TextPrimitives, float_parses, int_parses, parse_float, parse_int
from udfx.semantics.padding import PaddingPrimitives, pad_text
from udfx.semantics.regex import RegexPrimitives, lower_regex
from udfx.semantics.text import PADDING_METHODS, PREDICATES, string_predicate
from udfx.semantics.text_predicates import AsciiPrimitives, ascii_predicate
from udfx.semantics.translation import lower_translation
from udfx.types import Kind


def register_regex[E](rules: RuleSet[E], primitives: RegexPrimitives) -> None:
    def rule(name: str) -> Rule[E]:
        return lambda *arguments: Lowered(*lower_regex(primitives, name, *arguments))

    for name in ("search", "match", "fullmatch", "findall"):
        rules(f"re.{name}", Kind.STR, Kind.STR, Kind.INT, literals=(1, 2))(rule(name))
    rules("re.sub", Kind.STR, Kind.STR, Kind.INT, Kind.STR, Kind.INT, literals=(1, 2, 3, 4))(
        lambda text, pattern, flags, substitute, count: Lowered(
            *lower_regex(
                primitives, "sub", text, pattern, flags, substitute=substitute, count=count
            )
        )
    )
    rules("re.split", Kind.STR, Kind.STR, Kind.INT, Kind.INT, literals=(1, 2, 3))(
        lambda text, pattern, flags, count: Lowered(
            *lower_regex(primitives, "split", text, pattern, flags, count=count)
        )
    )


def register_string_predicates[E](
    rules: RuleSet[E], matches: Callable[[E, str], E], ascii: AsciiPrimitives | None = None
) -> None:
    for method in PREDICATES:
        rules(f".{method}", Kind.STR)(
            lambda value, method=method: (
                string_predicate(matches, value, method)
                if ascii is None
                else ascii_predicate(ascii, matches, value, method)
            )
        )


def register_translation[E](rules: RuleSet[E], primitives: TextPrimitives) -> None:
    rules(".translate", Kind.STR, Kind.STR, Kind.LIST, literals=(1, 2))(
        lambda text, characters, replacements: Lowered(
            *lower_translation(primitives, text, characters, replacements)
        )
    )


def register_padding[E](rules: RuleSet[E], primitives: PaddingPrimitives) -> None:
    for method in PADDING_METHODS:
        rules(f".{method}", Kind.STR, Kind.INT)(
            lambda value, width, method=method: Lowered(
                *pad_text(primitives, value, width, primitives.literal(" "), method)
            )
        )
        if method != "zfill":
            rules(f".{method}", Kind.STR, Kind.INT, Kind.STR)(
                lambda value, width, fill, method=method: Lowered(
                    *pad_text(primitives, value, width, fill, method)
                )
            )


def register_text_conversions[E](rules: RuleSet[E], text: TextPrimitives) -> None:
    def register(name: str, parse: Callable[[TextPrimitives, Any], tuple[Any, Any]]) -> None:
        rules(name, Kind.STR)(lambda value: Lowered(*parse(text, value)))

    rules("int_parses", Kind.STR, facts={"maximum_length": argument_length})(
        lambda value, maximum_length=None: Lowered(*int_parses(text, value, maximum_length))
    )

    def integer(value: E, maximum_length: int | None = None) -> Lowered[E]:
        parsed, exact = parse_int(text, value, maximum_length)
        return Lowered(parsed, overflow=None if exact is None else ~exact)

    rules("int", Kind.STR, facts={"maximum_length": argument_length})(integer)
    register("float_parses", float_parses)
    register("float", parse_float)
