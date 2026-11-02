from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import reduce

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import Evaluation, single
from udfx.frontend.core.expressions import invoke, node_of
from udfx.frontend.core.logic import choose, conjunction, disjunction, negate
from udfx.frontend.core.series import AGGREGATE
from udfx.frontend.core.tracing import AGGREGATES
from udfx.frontend.core.values import Item, Method, Opaque, Series, Value
from udfx.frontend.resolve.callables import known_op
from udfx.ir import FALSE, TRUE, Col, Guarded, If, Lit, Node, lit
from udfx.program import Reduction
from udfx.types import FLOAT, INT, INT64_MAX, INT64_MIN, Kind

ARITHMETIC = {"+": "__add__", "-": "__sub__", "*": "__mul__", "/": "__truediv__", "**": "__pow__"}
COMPARISONS = {
    "<": "__lt__",
    "<=": "__le__",
    "==": "__eq__",
    "!=": "__ne__",
    ">": "__gt__",
    ">=": "__ge__",
}
WRAPPING = {"+": "numpy.add", "-": "numpy.subtract", "*": "numpy.multiply"}
METHODS = {"add": "+", "sub": "-", "mul": "*", "div": "/", "truediv": "/", "pow": "**"}
EXTREMES = {"numpy.minimum": "min", "numpy.maximum": "max"}
NUMBERS = (Kind.INT, Kind.FLOAT)
MAX_EXPONENT = 8
REDUCTIONS = {"count", "sum", "mean", "var", "std", "max", "min", "corr"}


def operand(item: Item) -> Node:
    node = item.node if isinstance(item, Series) else node_of(item)
    if node.type.nullable or node.type.kind not in (*NUMBERS, Kind.STR, Kind.BOOL):
        raise Unsupported("series operands must be non-null scalars")
    if isinstance(node, Lit) and node.type.kind is Kind.INT:
        value = node.value
        if not isinstance(value, int) or not INT64_MIN <= value <= INT64_MAX:
            raise Unsupported("series integer literals must fit int64")
    return node


def elementwise(
    name: str, items: Sequence[Item], arguments: Sequence[Node], *, align: bool = False
) -> Evaluation:
    presents = list(dict.fromkeys(item.present for item in items if isinstance(item, Series)))
    node, faults = invoke(name, arguments, TypeError)
    inexact = reduce(disjunction, (fault.when for fault in faults), FALSE)
    value = node if inexact == FALSE else Guarded(node, negate(inexact))
    if len(presents) == 1:
        return single(Series(value, presents[0]))
    if not align or node.type.kind is not Kind.FLOAT:
        raise Unsupported("series with different rows require float arithmetic")
    shared = reduce(conjunction, presents)
    return single(Series(If(shared, value, lit(math.nan), FLOAT), reduce(disjunction, presents)))


def series_binary(symbol: str, left: Item, right: Item) -> Evaluation | None:
    if not isinstance(left, Series) and not isinstance(right, Series):
        return None
    items = (left, right)
    first, second = operand(left), operand(right)
    kinds = {first.type.kind, second.type.kind}
    if symbol in ("&", "|") and kinds == {Kind.BOOL}:
        return elementwise("and" if symbol == "&" else "or", items, [first, second])
    if kinds == {Kind.STR} and symbol == "+":
        return elementwise("__add__", items, [first, second])
    if symbol not in ARITHMETIC or not kinds <= set(NUMBERS):
        raise Unsupported(f"series operator {symbol}")
    if symbol == "/":
        first, second = (
            invoke("float", [node], TypeError)[0] if node.type.kind is Kind.INT else node
            for node in (first, second)
        )
    if symbol == "**" and kinds == {Kind.INT}:
        return integer_power(items, first, second)
    if symbol in WRAPPING and kinds == {Kind.INT}:
        return elementwise(WRAPPING[symbol], items, [first, second])
    return elementwise(ARITHMETIC[symbol], items, [first, second], align=True)


def integer_power(items: Sequence[Item], base: Node, exponent: Node) -> Evaluation:
    if not (isinstance(exponent, Lit) and isinstance(exponent.value, int)):
        raise Unsupported("series integer powers require a literal exponent")
    if not 0 <= exponent.value <= MAX_EXPONENT:
        raise Unsupported(f"series integer powers require an exponent from 0 to {MAX_EXPONENT}")
    present = next(item.present for item in items if isinstance(item, Series))
    return single(Series(reduce(multiply, [base] * exponent.value, lit(1)), present))


def multiply(left: Node, right: Node) -> Node:
    return right if left == lit(1) else invoke("numpy.multiply", [left, right], TypeError)[0]


def series_comparison(symbol: str, left: Item, right: Item) -> Evaluation | None:
    if not isinstance(left, Series) and not isinstance(right, Series):
        return None
    first, second = operand(left), operand(right)
    kinds = {first.type.kind, second.type.kind}
    if symbol not in COMPARISONS or (len(kinds) > 1 and kinds != set(NUMBERS)):
        raise Unsupported(f"series comparison {symbol}")
    if kinds == set(NUMBERS) and not any(
        isinstance(node, Lit) and isinstance(node.value, int) and abs(node.value) <= 2**53
        for node in (first, second)
    ):
        raise Unsupported("series comparisons of int and float require a small int literal")
    return elementwise(COMPARISONS[symbol], (left, right), [first, second])


def series_negation(item: Item) -> Evaluation | None:
    if not isinstance(item, Series):
        return None
    node = operand(item)
    name = "numpy.negative" if node.type.kind is Kind.INT else "__neg__"
    return elementwise(name, (item,), [node])


def series_filter(container: Item, index: Item) -> Evaluation | None:
    if not isinstance(container, Series):
        return None
    if AGGREGATES.get() is None or not isinstance(index, Series):
        raise Unsupported("series indexing requires a boolean series in an aggregate UDF")
    if operand(index).type.kind is not Kind.BOOL or index.present not in (TRUE, container.present):
        raise Unsupported("series masks require aligned boolean series")
    return single(Series(container.node, conjunction(container.present, index.node)))


def reduction(method: str, sources: Sequence[Item]) -> Evaluation:
    found = AGGREGATES.get()
    if found is None:
        raise Unsupported(f"series {method} requires an aggregate UDF")
    series = [source for source in sources if isinstance(source, Series)]
    if len(series) != len(sources):
        raise Unsupported(f"series {method} requires series arguments")
    nodes = tuple(operand(source) for source in series)
    kinds = {node.type.kind for node in nodes}
    numeric = kinds <= set(NUMBERS)
    if method == "count" and kinds != {Kind.FLOAT}:
        method = "len"
    if method in ("len", "count") and len(nodes) == 1:
        result = INT
    elif method == "sum" and len(nodes) == 1 and numeric:
        result = nodes[0].type
    elif (
        (method in ("mean", "var", "std") and len(nodes) == 1 and numeric)
        or (method in ("max", "min") and len(nodes) == 1 and kinds == {Kind.FLOAT})
        or (method == "corr" and len(nodes) == 2 and numeric)
    ):
        result = FLOAT
    else:
        raise Unsupported(f"series {method} of {', '.join(sorted(kind.value for kind in kinds))}")
    present = reduce(conjunction, (source.present for source in series))
    found.append(Reduction(method, nodes, present, result))
    return single(Value(Col(f"{AGGREGATE}{len(found) - 1}", result)))


def mapped(series: Series, table: object) -> Evaluation:
    node = operand(series)
    if not (
        isinstance(table, dict)
        and node.type.kind is Kind.STR
        and all(type(key) is str and type(value) is float for key, value in table.items())
    ):
        raise Unsupported("series map requires a dict from str to float")
    cases = [
        (invoke("__eq__", [node, lit(key)], TypeError)[0], lit(value))
        for key, value in table.items()
    ]
    return single(Series(choose([*cases, (TRUE, lit(math.nan))]), series.present))


def series_call(
    callee: Item, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> Evaluation | None:
    if not any(isinstance(argument, Series) for argument in arguments):
        return None
    method = callee.name if isinstance(callee, Method) else None
    name = (
        "abs" if method == "abs" else known_op(callee.value) if isinstance(callee, Opaque) else None
    )
    if method is None and not (name in ("abs", "len") or (name or "").startswith("numpy.")):
        return None
    if keywords:
        raise Unsupported("series calls with keywords")
    receiver = arguments[0]
    if method in METHODS and len(arguments) == 2:
        return series_binary(METHODS[method], receiver, arguments[1])
    if method in REDUCTIONS:
        return reduction(method, arguments)
    if method == "map" and isinstance(receiver, Series) and len(arguments) == 2:
        table = arguments[1]
        return mapped(receiver, table.value if isinstance(table, Opaque) else None)
    if name == "len" and len(arguments) == 1:
        return reduction("len", arguments)
    if name == "numpy.power" and len(arguments) == 2:
        return series_binary("**", receiver, arguments[1])
    if name is None or not (name == "abs" or name.startswith("numpy.")):
        return None
    nodes = [operand(argument) for argument in arguments]
    kinds = {node.type.kind for node in nodes}
    if name in EXTREMES and len(nodes) == 2 and kinds == {Kind.INT}:
        return elementwise(EXTREMES[name], arguments, nodes)
    if name in ("abs", "numpy.abs") and len(nodes) == 1 and kinds <= set(NUMBERS):
        return elementwise("numpy.absolute" if kinds == {Kind.INT} else "abs", arguments, nodes)
    if len(nodes) == 1:
        return elementwise(name, arguments, nodes)
    raise Unsupported(f"series call {name}")
