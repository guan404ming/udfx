from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from udfx.backends.rules import BinaryRule, Lowered, RuleSet
from udfx.semantics.numbers.math import (
    guarded_atan2,
    guarded_trigonometric,
    transcendental,
    trigonometric_argument,
)
from udfx.semantics.numbers.operations import RoundPrimitives, numpy_root, signed_round
from udfx.types import Kind


def register_float_variants[E](
    rules: RuleSet[E],
    name: str,
    rule: BinaryRule[E],
    to_float: Callable[[E], E],
    literals: tuple[int, ...] = (),
) -> None:
    rules(name, Kind.FLOAT, Kind.FLOAT, literals=literals)(rule)
    rules(name, Kind.INT, Kind.FLOAT, literals=literals)(
        lambda left, right: rule(to_float(left), right)
    )
    rules(name, Kind.FLOAT, Kind.INT, literals=literals)(
        lambda left, right: rule(
            left, cast("E", float(right)) if isinstance(right, int) else to_float(right)
        )
    )


def register_trigonometry[E](
    rules: RuleSet[E],
    rounding: RoundPrimitives,
    *,
    unary: Mapping[str, Callable[[E], E]],
    atan2: Callable[[E, E], E],
    is_nan: Callable[[E], E],
    guarded: bool = False,
) -> None:
    def angle(name: str, value: E) -> Lowered[E]:
        argument = trigonometric_argument(rounding, is_nan, name, value)
        native = unary[name](argument)
        return (
            Lowered(*guarded_trigonometric(rounding, is_nan, name, value, native))
            if guarded
            else Lowered(native)
        )

    for name in unary:
        rules(f"math.{name}", Kind.FLOAT)(lambda value, name=name: angle(name, value))
        rules(f"math.{name}", Kind.INT)(
            lambda value, name=name: angle(name, rounding.to_float(value))
        )

    def direction(vertical: E, horizontal: E) -> Lowered[E]:
        return (
            Lowered(*guarded_atan2(rounding, is_nan, atan2, vertical, horizontal))
            if guarded
            else Lowered(atan2(vertical, horizontal))
        )

    register_float_variants(rules, "math.atan2", direction, rounding.to_float)
    rules("math.atan2", Kind.INT, Kind.INT)(
        lambda vertical, horizontal: direction(
            rounding.to_float(vertical), rounding.to_float(horizontal)
        )
    )


def register_math[E](
    rules: RuleSet[E],
    rounding: RoundPrimitives,
    *,
    root: Callable[[E], E],
    floor: Callable[[E], E],
    ceil: Callable[[E], E],
    is_nan: Callable[[E], E],
    round_digits: BinaryRule[E],
    numpy_values: Callable[[str, E], E | None] | None = None,
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

            def rule(value: E, name: str = name, module: str = module) -> Lowered[E]:
                found = (
                    None if module != "numpy" or numpy_values is None else numpy_values(name, value)
                )
                if found is not None:
                    return Lowered(found)
                return Lowered(*transcendental(rounding, is_nan, name, value))

            rules(f"{module}.{name}", Kind.FLOAT)(rule)
            rules(f"{module}.{name}", Kind.INT)(
                lambda value, rule=rule: rule(rounding.to_float(value))
            )

    def logarithm(value: E, base: E) -> Lowered[E]:
        return Lowered(value, False)

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
