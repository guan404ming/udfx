from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import Evaluation, single
from udfx.frontend.core.expressions import invoke
from udfx.frontend.core.logic import negate
from udfx.frontend.core.values import Item, Method, Opaque, Series, Value
from udfx.frontend.operations.series import (
    COMPARISONS,
    METHODS,
    NUMBERS,
    operand,
    series_binary,
    series_comparison,
    settled,
)
from udfx.frontend.operations.series_text import bound, text_method, value
from udfx.frontend.resolve.callables import known_op
from udfx.frontend.resolve.constants import constant_value
from udfx.ir import If, Lit, Node, lit
from udfx.types import FLOAT, STR, Kind

COMPARISON_METHODS = {name.strip("_"): symbol for symbol, name in COMPARISONS.items()}
REVERSED = {f"r{name}": symbol for name, symbol in METHODS.items()}
MISSING = {"isna": False, "isnull": False, "notna": True, "notnull": True}
TYPES = {float: FLOAT, "float": FLOAT, "float64": FLOAT, str: STR, "str": STR}

type Handler = Callable[[str, Series, Node, Sequence[Item], Mapping[str, Item]], Evaluation]


def scalar(item: Item, kind: Kind) -> Node:
    node = operand(item)
    if node.type.kind is kind:
        return node
    if kind is Kind.FLOAT and isinstance(node, Lit) and node.type.kind is Kind.INT:
        return lit(float(node.value))  # pyright: ignore[reportArgumentType]
    raise Unsupported(f"series {kind.value} methods need {kind.value} values")


def missing(node: Node) -> Node:
    return (
        invoke("numpy.isnan", [node], TypeError)[0] if node.type.kind is Kind.FLOAT else lit(False)
    )


def clipped(items: Sequence[Item], node: Node, lower: object, upper: object) -> Evaluation:
    limits = [limit for limit in (lower, upper) if limit is not None]
    if node.type.kind not in NUMBERS or not all(
        isinstance(limit, int | float) and type(limit) is not bool and math.isfinite(limit)
        for limit in limits
    ):
        raise Unsupported("series clip needs numbers and finite literal bounds")
    if len(limits) == 2:
        lower, upper = sorted(limits)  # pyright: ignore[reportArgumentType]
    result = node
    for limit, symbol in ((lower, "__lt__"), (upper, "__gt__")):
        if limit is not None:
            edge = scalar(Value(lit(limit)), node.type.kind)
            result = If(invoke(symbol, [result, edge], TypeError)[0], edge, result, node.type)
    return settled(items, result, [])


def selected(
    items: Sequence[Item], condition: Item, then: Item, orelse: Item, *, widen: bool
) -> Evaluation:
    test = operand(condition)
    if test.type.kind is not Kind.BOOL:
        raise Unsupported("series selection needs a boolean condition")
    first, second = operand(then), operand(orelse)
    if widen and {first.type.kind, second.type.kind} == set(NUMBERS):
        first, second = (
            invoke("float", [part], TypeError)[0] if part.type.kind is Kind.INT else part
            for part in (first, second)
        )
    if first.type != second.type:
        raise Unsupported("series selection needs values of one type")
    return settled(items, If(test, first, second, first.type), [])


def paired(
    name: str, receiver: Series, node: Node, rest: Sequence[Item], keywords: Mapping[str, Item]
) -> Evaluation:
    (other,) = bound(rest, keywords, ("other",)).values()
    result = (
        series_comparison(COMPARISON_METHODS[name], receiver, other)
        if name in COMPARISON_METHODS
        else series_binary(REVERSED[name], other, receiver)
    )
    if result is None:
        raise Unsupported(f"series {name}")
    return result


def tested(
    name: str, receiver: Series, node: Node, rest: Sequence[Item], keywords: Mapping[str, Item]
) -> Evaluation:
    bound(rest, keywords, ())
    found = missing(node)
    return settled([receiver], negate(found) if MISSING[name] else found, [])


def filled(
    name: str, receiver: Series, node: Node, rest: Sequence[Item], keywords: Mapping[str, Item]
) -> Evaluation:
    (other,) = bound(rest, keywords, ("value",)).values()
    if node.type.kind is not Kind.FLOAT:
        return single(receiver)
    return settled([receiver], If(missing(node), scalar(other, Kind.FLOAT), node, FLOAT), [])


def rounded(
    name: str, receiver: Series, node: Node, rest: Sequence[Item], keywords: Mapping[str, Item]
) -> Evaluation:
    decimals = value(bound(rest, keywords, ("decimals",)), "decimals", 0)
    if type(decimals) is not int or (node.type.kind is Kind.INT and decimals < 0):
        raise Unsupported("series round needs literal integer decimals")
    if node.type.kind is Kind.INT:
        return single(receiver)
    nodes = [node] if decimals == 0 else [node, lit(decimals)]
    return settled([receiver], *invoke("numpy.round", nodes, TypeError))


def clip(
    name: str, receiver: Series, node: Node, rest: Sequence[Item], keywords: Mapping[str, Item]
) -> Evaluation:
    found = bound(rest, keywords, ("lower", "upper"))
    return clipped([receiver], node, value(found, "lower"), value(found, "upper"))


def between(
    name: str, receiver: Series, node: Node, rest: Sequence[Item], keywords: Mapping[str, Item]
) -> Evaluation:
    found = bound(rest, keywords, ("left", "right", "inclusive"))
    inclusive = value(found, "inclusive", "both")
    if inclusive not in {"both", "neither", "left", "right"}:
        raise Unsupported("series between needs a literal inclusive")
    low = "__ge__" if inclusive in {"both", "left"} else "__gt__"
    high = "__le__" if inclusive in {"both", "right"} else "__lt__"
    left = invoke(low, [node, scalar(found["left"], node.type.kind)], TypeError)[0]
    right = invoke(high, [node, scalar(found["right"], node.type.kind)], TypeError)[0]
    return settled([receiver], invoke("and", [left, right], TypeError)[0], [])


def masked(
    name: str, receiver: Series, node: Node, rest: Sequence[Item], keywords: Mapping[str, Item]
) -> Evaluation:
    found = bound(rest, keywords, ("cond", "other"))
    if "other" not in found and node.type.kind is not Kind.FLOAT:
        raise Unsupported(f"series {name} without other needs floats")
    other = Value(scalar(found.get("other", Value(lit(math.nan))), node.type.kind))
    condition = found["cond"]
    kept, replaced = (receiver, other) if name == "where" else (other, receiver)
    return selected([receiver, condition], condition, kept, replaced, widen=False)


def converted(
    name: str, receiver: Series, node: Node, rest: Sequence[Item], keywords: Mapping[str, Item]
) -> Evaluation:
    (dtype,) = bound(rest, keywords, ("dtype",)).values()
    target = constant_value(dtype)
    found = next((kind for key, kind in TYPES.items() if key is target or key == target), None)
    if found is None:
        raise Unsupported("series astype supports only float and str")
    if found.kind is node.type.kind:
        return single(receiver)
    return settled([receiver], *invoke(found.kind.value, [node], TypeError))


HANDLERS: dict[str, Handler] = {
    **dict.fromkeys(COMPARISON_METHODS | REVERSED, paired),
    **dict.fromkeys(MISSING, tested),
    "fillna": filled,
    "round": rounded,
    "clip": clip,
    "between": between,
    "where": masked,
    "mask": masked,
    "astype": converted,
}


def library_method(name: str | None, arguments: Sequence[Item]) -> Evaluation | None:
    if name == "numpy.where" and len(arguments) == 3:
        return selected(arguments, *arguments, widen=True)
    if name == "numpy.clip" and len(arguments) == 3 and isinstance(arguments[0], Series):
        lower, upper = (constant_value(argument) for argument in arguments[1:])
        return clipped(arguments[:1], operand(arguments[0]), lower, upper)
    return None


def series_method(
    callee: Item, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> Evaluation | None:
    if isinstance(callee, Opaque):
        name = known_op(callee.value)
        if name in {"pandas.isna", "pandas.notna"} and len(arguments) == 1:
            callee = Method(name.removeprefix("pandas."))
        elif keywords:
            return None
        else:
            return library_method(name, arguments)
    if not isinstance(callee, Method) or not arguments or not isinstance(arguments[0], Series):
        return None
    receiver, rest = arguments[0], arguments[1:]
    if receiver.text:
        return text_method(receiver, callee.name, rest, keywords)
    handler = HANDLERS.get(callee.name)
    if handler is None:
        return None
    return handler(callee.name, receiver, operand(receiver), rest, keywords)
