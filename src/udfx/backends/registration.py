from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from udfx.ops.registry import OpKey
from udfx.semantics.datetimes import (
    ATTRIBUTES,
    DatetimeArithmeticPrimitives,
    DatetimePrimitives,
    datetime_field,
    format_datetime,
    parse_datetime,
    shift_datetime,
    shift_overflows,
)
from udfx.semantics.lists import (
    ListPrimitives,
    SetPrimitives,
    native_range,
    native_set,
    repeat_list,
)
from udfx.semantics.literals import (
    TextPrimitives,
    float_parses,
    int_parses,
    parse_float,
    parse_int,
)
from udfx.semantics.math import transcendental
from udfx.semantics.numbers import (
    COMPARISONS,
    Binary,
    OrderPrimitives,
    RoundPrimitives,
    float_compare,
    float_extremum,
    numpy_root,
    signed_round,
    widened_exact,
)
from udfx.semantics.regex import RegexPrimitives, lower_regex
from udfx.semantics.scalars import missing_expression
from udfx.semantics.text import (
    PADDING_METHODS,
    PREDICATES,
    PaddingPrimitives,
    pad_text,
    string_predicate,
)
from udfx.semantics.translation import lower_translation
from udfx.types import Kind


@dataclass(frozen=True)
class Lowered[E]:
    expr: E
    exact: E | None = None


type RuleResult[E] = E | Lowered[E]
type Rule[E] = Callable[..., RuleResult[E]]
type BinaryRule[E] = Callable[[E, E], RuleResult[E]]


BOOLEAN_RULES: dict[OpKey, Rule[Any]] = {
    OpKey("not", (Kind.BOOL,)): operator.invert,
    OpKey("and", (Kind.BOOL, Kind.BOOL)): operator.and_,
    OpKey("or", (Kind.BOOL, Kind.BOOL)): operator.or_,
}


class RuleSet[E]:
    def __init__(self) -> None:
        self._rules: dict[OpKey, Rule[E]] = dict(BOOLEAN_RULES)

    def __call__[**P](
        self, name: str, *kinds: Kind
    ) -> Callable[[Callable[P, RuleResult[E]]], Callable[P, RuleResult[E]]]:
        key = OpKey(name, kinds)

        def decorate(rule: Callable[P, RuleResult[E]]) -> Callable[P, RuleResult[E]]:
            if key in self._rules:
                raise ValueError(f"duplicate rule {key}")
            self._rules[key] = rule
            return rule

        return decorate

    def get(self, key: OpKey) -> Rule[E] | None:
        return self._rules.get(key)

    def keys(self) -> frozenset[OpKey]:
        return frozenset(self._rules)


def register_regex[E](rules: RuleSet[E], primitives: RegexPrimitives) -> None:
    def rule(name: str) -> Rule[E]:
        return lambda *arguments: Lowered(*lower_regex(primitives, name, *arguments))

    for name in ("search", "match", "fullmatch", "findall"):
        rules(f"re.{name}", Kind.STR, Kind.STR, Kind.INT)(rule(name))
    rules("re.sub", Kind.STR, Kind.STR, Kind.INT, Kind.STR, Kind.INT)(
        lambda text, pattern, flags, substitute, count: Lowered(
            *lower_regex(
                primitives, "sub", text, pattern, flags, substitute=substitute, count=count
            )
        )
    )
    rules("re.split", Kind.STR, Kind.STR, Kind.INT, Kind.INT)(
        lambda text, pattern, flags, count: Lowered(
            *lower_regex(primitives, "split", text, pattern, flags, count=count)
        )
    )


def register_string_predicates[E](rules: RuleSet[E], matches: Callable[[E, str], E]) -> None:
    for method in PREDICATES:
        rules(f".{method}", Kind.STR)(
            lambda value, method=method: string_predicate(matches, value, method)
        )


def register_translation[E](rules: RuleSet[E], primitives: TextPrimitives) -> None:
    rules(".translate", Kind.STR, Kind.STR, Kind.LIST)(
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


def register_float_variants[E](
    rules: RuleSet[E], name: str, rule: BinaryRule[E], to_float: Callable[[E], E]
) -> None:
    rules(name, Kind.FLOAT, Kind.FLOAT)(rule)
    rules(name, Kind.INT, Kind.FLOAT)(lambda left, right: rule(to_float(left), right))
    rules(name, Kind.FLOAT, Kind.INT)(lambda left, right: rule(left, to_float(right)))


def register_missing[E](
    rules: RuleSet[E], is_null: Callable[[E], Any], is_nan: Callable[[E], Any]
) -> None:
    def missing(value: E) -> Any:
        return missing_expression(is_null, is_nan, value)

    rules("pandas.isna", Kind.FLOAT)(missing)
    rules("pandas.notna", Kind.FLOAT)(lambda value: ~missing(value))
    for kind in (Kind.STR, Kind.INT, Kind.BOOL):
        rules("pandas.isna", kind)(is_null)
        rules("pandas.notna", kind)(lambda value: ~is_null(value))


def register_math[E](
    rules: RuleSet[E],
    rounding: RoundPrimitives,
    *,
    root: Callable[[E], E],
    floor: Callable[[E], E],
    ceil: Callable[[E], E],
    is_nan: Callable[[E], E],
    round_digits: BinaryRule[E],
) -> None:
    def square_root(value: E) -> E:
        return numpy_root(rounding, root, value)

    rules("numpy.sqrt", Kind.FLOAT)(square_root)
    rules("numpy.sqrt", Kind.INT)(lambda value: square_root(rounding.to_float(value)))
    for module, names in (
        ("numpy", ("log", "log1p", "log2", "log10", "exp", "expm1")),
        ("math", ("log", "log2", "exp")),
    ):
        for name in names:

            def rule(value: E, name: str = name) -> Lowered[E]:
                return Lowered(*transcendental(rounding, is_nan, name, value))

            rules(f"{module}.{name}", Kind.FLOAT)(rule)
            rules(f"{module}.{name}", Kind.INT)(
                lambda value, rule=rule: rule(rounding.to_float(value))
            )

    def logarithm(value: E, base: E) -> Lowered[E]:
        return Lowered(value, rounding.literal(False))

    register_float_variants(rules, "math.log", logarithm, rounding.to_float)
    rules("math.log", Kind.INT, Kind.INT)(
        lambda value, base: logarithm(rounding.to_float(value), base)
    )
    rules("numpy.isnan", Kind.FLOAT)(is_nan)
    for kind in (Kind.INT, Kind.BOOL):
        rules("numpy.isnan", kind)(lambda value: rounding.literal(False))
    rules("numpy.abs", Kind.FLOAT)(rounding.abs)
    rules("numpy.floor", Kind.FLOAT)(floor)
    rules("numpy.ceil", Kind.FLOAT)(ceil)
    rules("numpy.round", Kind.FLOAT)(lambda value: signed_round(rounding, value))
    rules("numpy.round", Kind.FLOAT, Kind.INT)(round_digits)


def register_widened_variants[E](
    rules: RuleSet[E], name: str, rule: Binary, to_float: Callable[[E], E], order: OrderPrimitives
) -> None:
    rules(name, Kind.INT, Kind.FLOAT)(
        lambda left, right: Lowered(rule(to_float(left), right), widened_exact(order, left))
    )
    rules(name, Kind.FLOAT, Kind.INT)(
        lambda left, right: Lowered(rule(left, to_float(right)), widened_exact(order, right))
    )


def register_comparison[E](
    rules: RuleSet[E],
    name: str,
    compare: Binary,
    to_float: Callable[[E], E],
    order: OrderPrimitives,
) -> None:
    def ordered(left: E, right: E) -> E:
        return float_compare(order, compare, left, right)

    rules(name, Kind.INT, Kind.INT)(compare)
    rules(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    register_widened_variants(rules, name, ordered, to_float, order)


def register_equality[E](
    rules: RuleSet[E], equal: Binary, to_float: Callable[[E], E], order: OrderPrimitives
) -> None:
    def unequal(left: Any, right: Any) -> Any:
        return ~equal(left, right)

    def float_unequal(left: Any, right: Any) -> Any:
        return ~float_compare(order, equal, left, right)

    register_comparison(rules, "__eq__", equal, to_float, order)
    rules("__ne__", Kind.INT, Kind.INT)(unequal)
    rules("__ne__", Kind.FLOAT, Kind.FLOAT)(float_unequal)
    register_widened_variants(rules, "__ne__", float_unequal, to_float, order)


def register_extremum[E](
    rules: RuleSet[E], name: str, pick: Binary, order: OrderPrimitives
) -> None:
    rules(name, Kind.FLOAT, Kind.FLOAT)(
        lambda left, right: Lowered(*float_extremum(order, pick, left, right))
    )
    for kind in (Kind.INT, Kind.STR):
        rules(name, kind, kind)(pick)


def register_text_conversions[E](rules: RuleSet[E], text: TextPrimitives) -> None:
    def register(name: str, parse: Callable[[TextPrimitives, Any], tuple[Any, Any]]) -> None:
        rules(name, Kind.STR)(lambda value: Lowered(*parse(text, value)))

    rules("int_parses", Kind.STR)(
        lambda value, maximum_length=None: Lowered(*int_parses(text, value, maximum_length))
    )
    rules("int", Kind.STR)(
        lambda value, maximum_length=None: Lowered(*parse_int(text, value, maximum_length))
    )
    register("float_parses", float_parses)
    register("float", parse_float)


def register_datetimes[E](rules: RuleSet[E], primitives: DatetimePrimitives, equal: Binary) -> None:
    for name in (*ATTRIBUTES, "weekday", "isoweekday", "toordinal"):
        rules(f".{name}", Kind.DATETIME)(
            lambda value, name=name: datetime_field(primitives, value, name)
        )
    for name, compare in COMPARISONS.items():
        rules(name, Kind.DATETIME, Kind.DATETIME)(compare)
    rules("__eq__", Kind.DATETIME, Kind.DATETIME)(equal)
    rules("__ne__", Kind.DATETIME, Kind.DATETIME)(lambda left, right: ~equal(left, right))
    rules("bool", Kind.DATETIME)(lambda value: ~primitives.is_null(value))
    rules(".strftime", Kind.DATETIME, Kind.STR)(
        lambda value, format: Lowered(*format_datetime(primitives, value, format))
    )
    rules("datetime.strptime", Kind.STR, Kind.STR)(
        lambda text, format: Lowered(*parse_datetime(primitives, text, format))
    )

    def parses(text: E, format: Any) -> Lowered[E]:
        parsed, exact = parse_datetime(primitives, text, format)
        return Lowered(~primitives.is_null(parsed), exact)

    rules("datetime_parses", Kind.STR, Kind.STR)(parses)


def register_datetime_arithmetic[E](
    rules: RuleSet[E], primitives: DatetimeArithmeticPrimitives
) -> None:
    rules("datetime_difference", Kind.DATETIME, Kind.DATETIME)(
        lambda left, right: primitives.to_microseconds(left) - primitives.to_microseconds(right)
    )
    rules("datetime_shift", Kind.DATETIME, Kind.INT)(
        lambda value, offset: shift_datetime(primitives, value, offset)
    )
    rules("datetime_shift_overflows", Kind.DATETIME, Kind.INT)(
        lambda value, offset: shift_overflows(primitives, primitives.to_microseconds(value), offset)
    )


def register_lists[E](rules: RuleSet[E], primitives: ListPrimitives, sets: SetPrimitives) -> None:
    rules("__mul__", Kind.LIST, Kind.INT)(
        lambda value, count: Lowered(*repeat_list(primitives, value, count))
    )
    rules("__mul__", Kind.INT, Kind.LIST)(
        lambda count, value: Lowered(*repeat_list(primitives, value, count))
    )
    rules("range", Kind.INT)(
        lambda stop: Lowered(
            *native_range(primitives, primitives.literal(0), stop, primitives.literal(1))
        )
    )
    rules("range", Kind.INT, Kind.INT, Kind.INT)(
        lambda start, stop, step: Lowered(*native_range(primitives, start, stop, step))
    )
    for operation in ("set_size", "set_list", "sorted_set"):
        for floating in (False, True):
            rules(operation + ("_float" if floating else ""), Kind.LIST)(
                lambda value, operation=operation, floating=floating: Lowered(
                    *native_set(sets, value, operation, floating)
                )
            )
