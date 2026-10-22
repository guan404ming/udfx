from __future__ import annotations

from collections.abc import Callable
from typing import Any

from udfx.backends.registration_datetimes import register_datetime_arithmetic, register_datetimes
from udfx.backends.registration_text import (
    register_padding,
    register_regex,
    register_string_predicates,
    register_text_conversions,
    register_translation,
)
from udfx.backends.rules import BinaryRule, Lowered, Rule, RuleOptions, RuleResult, RuleSet
from udfx.semantics.lists import (
    SET_OPERATIONS,
    SET_PREDICATES,
    ListPrimitives,
    SetPrimitives,
    native_range,
    native_set,
    native_set_operation,
    repeat_list,
)
from udfx.semantics.math import transcendental
from udfx.semantics.numbers import (
    FLOAT_INT_MAX,
    Binary,
    OrderPrimitives,
    RoundPrimitives,
    float_compare,
    float_extremum,
    numpy_root,
    signed_round,
    widened_exact,
)
from udfx.semantics.scalars import missing_expression
from udfx.types import Kind

__all__ = [
    "BinaryRule",
    "Lowered",
    "Rule",
    "RuleOptions",
    "RuleResult",
    "RuleSet",
    "register_datetime_arithmetic",
    "register_datetimes",
    "register_padding",
    "register_regex",
    "register_string_predicates",
    "register_text_conversions",
    "register_translation",
]


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
    def exact(value: Any) -> Any:
        primitives = (
            OrderPrimitives(
                lambda left, right: False,
                lambda value: value is None,
                lambda value: value is not None and -FLOAT_INT_MAX <= value <= FLOAT_INT_MAX,
            )
            if isinstance(value, int) or value is None
            else order
        )
        return widened_exact(primitives, value)

    rules(name, Kind.INT, Kind.FLOAT, guards={0: exact})(
        lambda left, right: rule(to_float(left), right)
    )
    rules(name, Kind.FLOAT, Kind.INT, guards={1: exact})(
        lambda left, right: rule(left, to_float(right))
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


def register_lists[E](rules: RuleSet[E], primitives: ListPrimitives, sets: SetPrimitives) -> None:
    for operation in (*SET_OPERATIONS, *SET_PREDICATES):
        rules(f"set_{operation}", Kind.LIST, Kind.LIST)(
            lambda left, right, operation=operation: native_set_operation(
                sets, left, right, operation
            )
        )
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
