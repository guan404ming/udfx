from __future__ import annotations

from collections.abc import Callable
from typing import Any

from udfx.backends.guarantees import Guarantee
from udfx.backends.rules import Lowered, Rule, RuleSet
from udfx.ir import FALSE
from udfx.semantics.facts import (
    argument_length,
    argument_maximum,
    ascii_predicate_result,
    integer_is_valid,
    json_validity,
)
from udfx.semantics.json import lower_string, patterns
from udfx.semantics.literals import (
    TextPrimitives,
    float_parses,
    int_parses,
    integer_candidates,
    parse_float,
    parse_int,
    parse_saturated_int,
)
from udfx.semantics.padding import PaddingPrimitives, pad_text
from udfx.semantics.regex import RegexPrimitives, lower_regex
from udfx.semantics.text import PADDING_METHODS, PREDICATES, string_predicate
from udfx.semantics.text_bounds import argument_without_newline
from udfx.semantics.text_predicates import AsciiPrimitives, ascii_predicate
from udfx.semantics.translation import lower_translation
from udfx.types import Kind


def register_json[E](
    rules: RuleSet[E],
    primitives: TextPrimitives,
    concatenate: Callable[[E, E], E],
    findall: Callable[[E, str], E],
    *,
    possessive: bool = False,
    encode: Callable[[E], E] | None = None,
) -> None:
    string, array, scalar = patterns(possessive)
    for name, pattern in (("json.string_native", scalar), ("json.array_native", array)):
        rules(name, Kind.STR, facts={"validity": json_validity})(
            lambda value, validity, pattern=pattern: Lowered(
                primitives.matches(value, rf"\A{pattern}\z"),
                guarantees=() if validity is None else (Guarantee(validity, FALSE),),
            )
        )
    rules("json.tokens", Kind.STR)(
        lambda value: Lowered(findall(value, string), primitives.matches(value, rf"\A{array}\z"))
    )
    rules("json.dumps", Kind.STR, Kind.BOOL, literals=(1,))(
        lambda value, ensure_ascii: Lowered(
            *lower_string(primitives, concatenate, value, ensure_ascii, encode)
        )
    )


def register_regex[E](
    rules: RuleSet[E], primitives: RegexPrimitives, ascii: AsciiPrimitives | None = None
) -> None:
    def rule(name: str) -> Rule[E]:
        return lambda *arguments, newline_free=False: Lowered(
            *lower_regex(primitives, name, *arguments, ascii=ascii, newline_free=newline_free)
        )

    facts = {"newline_free": argument_without_newline}
    for name in ("search", "match", "fullmatch", "findall", "first", "contains", "count"):
        rules(f"re.{name}", Kind.STR, Kind.STR, Kind.INT, literals=(1, 2), facts=facts)(rule(name))
    rules(
        "re.sub",
        Kind.STR,
        Kind.STR,
        Kind.INT,
        Kind.STR,
        Kind.INT,
        literals=(1, 2, 3, 4),
        facts=facts,
    )(
        lambda text, pattern, flags, substitute, count, newline_free=False: Lowered(
            *lower_regex(
                primitives,
                "sub",
                text,
                pattern,
                flags,
                substitute=substitute,
                count=count,
                ascii=ascii,
                newline_free=newline_free,
            )
        )
    )
    rules("re.split", Kind.STR, Kind.STR, Kind.INT, Kind.INT, literals=(1, 2, 3), facts=facts)(
        lambda text, pattern, flags, count, newline_free=False: Lowered(
            *lower_regex(
                primitives,
                "split",
                text,
                pattern,
                flags,
                count=count,
                ascii=ascii,
                newline_free=newline_free,
            )
        )
    )


def register_string_predicates[E](
    rules: RuleSet[E], matches: Callable[[E, str], E], ascii: AsciiPrimitives | None = None
) -> None:
    for method in PREDICATES:
        rules(f".{method}", Kind.STR, facts={"known": ascii_predicate_result} if ascii else None)(
            lambda value, method=method, known=None: (
                string_predicate(matches, value, method)
                if ascii is None
                else ascii_predicate(ascii, matches, value, method, known)
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
        rules(f".{method}", Kind.STR, Kind.INT, facts={"maximum": argument_maximum})(
            lambda value, width, method=method, maximum=None: Lowered(
                *pad_text(
                    primitives, value, width, primitives.literal(" "), method, maximum=maximum
                )
            )
        )
        if method != "zfill":
            rules(f".{method}", Kind.STR, Kind.INT, Kind.STR, facts={"maximum": argument_maximum})(
                lambda value, width, fill, method=method, maximum=None: Lowered(
                    *pad_text(primitives, value, width, fill, method, maximum=maximum)
                )
            )


def register_text_conversions[E](rules: RuleSet[E], text: TextPrimitives) -> None:
    def register(name: str, parse: Callable[[TextPrimitives, Any], tuple[Any, Any]]) -> None:
        rules(name, Kind.STR)(lambda value: Lowered(*parse(text, value)))

    rules("int_parses", Kind.STR, facts={"maximum_length": argument_length})(
        lambda value, maximum_length=None: Lowered(*int_parses(text, value, maximum_length))
    )

    def integer(value: E, maximum_length: int | None = None, valid: bool = False) -> Lowered[E]:
        parsed, exact = parse_int(text, value, maximum_length, valid)
        return Lowered(
            parsed,
            overflow=None if exact is None else ~exact,
            recover_overflow=(
                (lambda failure: text.coalesce(*integer_candidates(text, value), failure))
                if valid and exact is not None
                else None
            ),
        )

    rules("int", Kind.STR, facts={"maximum_length": argument_length, "valid": integer_is_valid})(
        integer
    )
    rules("int_saturated", Kind.STR)(lambda value: parse_saturated_int(text, value))
    register("float_parses", float_parses)
    register("float", parse_float)
