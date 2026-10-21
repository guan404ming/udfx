from __future__ import annotations

import ast
import heapq
from collections.abc import Iterator

from bench.mine.models import Candidate
from udfx.semantics.datetimes import ATTRIBUTES
from udfx.types import BOOL, DATETIME, FLOAT, INT, STR, Type


def column_name(node: ast.AST, row_parameter: str | None) -> str | None:
    if row_parameter is None:
        return node.id if isinstance(node, ast.Name) else None
    if isinstance(node, ast.Attribute | ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id != row_parameter:
            return None
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            return node.slice.value
    return None


def schema_columns(candidate: Candidate) -> tuple[list[str], str | None]:
    arguments = candidate.node.args
    parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    parameters.extend(argument for argument in (arguments.vararg, arguments.kwarg) if argument)
    names = [parameter.arg for parameter in parameters]
    if candidate.kind != "row":
        return names, None
    if len(names) != 1:
        raise ValueError("row candidate requires one parameter")
    row_parameter = names[0]
    columns = list(
        dict.fromkeys(
            name
            for node in ast.walk(candidate.node)
            if (name := column_name(node, row_parameter)) is not None
        )
    )
    if not columns:
        raise ValueError("row columns could not be inferred")
    if columns == names:
        raise ValueError("row column names match parameter names")
    return columns, row_parameter


def schema_types(
    node: ast.FunctionDef | ast.Lambda, names: list[str], row_parameter: str | None
) -> list[list[Type]]:
    hints: dict[str, set[str]] = {name: set() for name in names}
    for child in ast.walk(node):
        expression: ast.AST = child
        hint = None
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr in {"strftime", "weekday", "isoweekday", "toordinal"}:
                expression = child.func.value
                hint = "datetime"
            elif child.func.attr in vars(str):
                expression = child.func.value
                hint = "str"
        elif isinstance(child, ast.Attribute) and child.attr in ATTRIBUTES:
            expression = child.value
            hint = "datetime"
        elif isinstance(child, ast.BinOp | ast.Compare):
            operands = (
                [child.left, child.right]
                if isinstance(child, ast.BinOp)
                else [child.left, *child.comparators]
            )
            if any(
                isinstance(value, ast.Constant) and isinstance(value.value, str)
                for value in operands
            ):
                hint = "str"
            elif isinstance(child, ast.BinOp):
                hint = "number"
        elif isinstance(child, ast.BoolOp) or (
            isinstance(child, ast.UnaryOp) and isinstance(child.op, ast.Not)
        ):
            hint = "bool"
        if hint is not None:
            for value in ast.walk(expression):
                name = column_name(value, row_parameter)
                if name in hints:
                    hints[name].add(hint)
    orders = {
        "datetime": (DATETIME, STR, FLOAT, INT, BOOL),
        "str": (STR, FLOAT, INT, BOOL),
        "number": (FLOAT, INT, STR, BOOL),
        "bool": (BOOL, STR, FLOAT, INT),
    }
    return [
        [
            column_type.or_none()
            for column_type in next(
                (order for hint, order in orders.items() if hint in hints[name]), orders["str"]
            )
        ]
        for name in names
    ]


def schemas(candidate: Candidate) -> Iterator[dict[str, Type]]:
    names, row_parameter = schema_columns(candidate)
    choices = schema_types(candidate.node, names, row_parameter)
    initial = (0,) * len(names)
    pending = [(0, initial)]
    visited = {initial}
    for _ in range(64):
        if not pending:
            return
        priority, indices = heapq.heappop(pending)
        yield {name: choices[index][indices[index]] for index, name in enumerate(names)}
        for index, value in enumerate(indices):
            if value == len(choices[index]) - 1:
                continue
            following = (*indices[:index], value + 1, *indices[index + 1 :])
            if following not in visited:
                visited.add(following)
                heapq.heappush(pending, (priority + 1, following))
