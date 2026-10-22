from __future__ import annotations

import ast
import heapq
from collections import Counter, deque
from collections.abc import Iterator

from bench.mine.bindings import HELPER_DEPTH
from bench.mine.models import Binding, Candidate
from udfx.semantics.datetimes import ATTRIBUTES
from udfx.types import BOOL, DATETIME, FLOAT, INT, STR, Type


def column_name(
    node: ast.AST, row_parameter: str | None, constants: dict[str, ast.expr]
) -> str | int | None:
    if row_parameter is None:
        return node.id if isinstance(node, ast.Name) else None
    if isinstance(node, ast.Attribute | ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id != row_parameter:
            return None
        if isinstance(node, ast.Attribute):
            return node.attr
        key = (
            constants.get(node.slice.id, node.slice)
            if isinstance(node.slice, ast.Name)
            else node.slice
        )
        try:
            value = ast.literal_eval(key)
        except (ValueError, TypeError, SyntaxError):
            return None
        if isinstance(value, str | int):
            return value
    return None


def function_nodes(node: ast.FunctionDef | ast.Lambda) -> list[ast.AST]:
    pending = deque(ast.iter_child_nodes(node))
    nodes: list[ast.AST] = []
    while pending:
        child = pending.popleft()
        nodes.append(child)
        if not isinstance(
            child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef
        ):
            pending.extend(ast.iter_child_nodes(child))
    return nodes


def schema_nodes(
    node: ast.FunctionDef | ast.Lambda,
    row_parameter: str | None,
    bindings: dict[str, Binding],
) -> Iterator[tuple[ast.AST, str | None, dict[str, ast.expr]]]:
    pending = deque([(node, row_parameter, bindings, -1)])
    visited: set[tuple[ast.AST, str | None]] = set()
    while pending:
        function, parameter, available, depth = pending.popleft()
        if (function, parameter) in visited or depth > HELPER_DEPTH:
            continue
        visited.add((function, parameter))
        nodes = function_nodes(function) if parameter is not None else list(ast.walk(function))
        writes = Counter(
            child.id
            for child in nodes
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store | ast.Del)
        )
        constants: dict[str, ast.expr] = {}
        local_names = {
            *writes,
            *(
                child.arg
                for child in ast.iter_child_nodes(function.args)
                if isinstance(child, ast.arg)
            ),
        }
        helpers = {name: binding for name, binding in available.items() if name not in local_names}
        local_helpers = {child.name: child for child in nodes if isinstance(child, ast.FunctionDef)}
        helpers.update(local_helpers)
        for child in nodes:
            if isinstance(child, ast.Assign | ast.AnnAssign) and child.value is not None:
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                for target in targets:
                    if isinstance(target, ast.Name) and writes[target.id] == 1:
                        constants[target.id] = child.value
        for child in nodes:
            yield child, parameter, constants
            if (
                parameter is None
                or not isinstance(child, ast.Call)
                or not isinstance(child.func, ast.Name)
            ):
                continue
            helper = helpers.get(child.func.id)
            if not isinstance(helper, ast.FunctionDef | ast.Lambda):
                continue
            scope = helpers if helper in local_helpers.values() else bindings
            parameters = [*helper.args.posonlyargs, *helper.args.args]
            arguments = [
                *(
                    (argument.arg, value)
                    for argument, value in zip(parameters, child.args, strict=False)
                ),
                *((keyword.arg, keyword.value) for keyword in child.keywords),
            ]
            for name, value in arguments:
                if name is not None and isinstance(value, ast.Name) and value.id == parameter:
                    pending.append((helper, name, scope, depth + 1))
            if helper in local_helpers.values() and not any(
                isinstance(argument, ast.arg) and argument.arg == parameter
                for argument in ast.iter_child_nodes(helper.args)
            ):
                pending.append((helper, parameter, scope, depth + 1))


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
    references = list(
        dict.fromkeys(
            name
            for node, parameter, constants in schema_nodes(
                candidate.node, row_parameter, candidate.bindings
            )
            if (name := column_name(node, parameter, constants)) is not None
        )
    )
    positions = [name for name in references if isinstance(name, int)]
    count = max((index + 1 if index >= 0 else -index for index in positions), default=0)
    columns = list(
        dict.fromkeys(
            [
                *(f"c{index}" for index in range(count)),
                *(name for name in references if isinstance(name, str)),
            ]
        )
    )
    if not columns:
        raise ValueError("row columns could not be inferred")
    if columns == names:
        raise ValueError("row column names match parameter names")
    return columns, row_parameter


def schema_types(
    node: ast.FunctionDef | ast.Lambda,
    names: list[str],
    row_parameter: str | None,
    bindings: dict[str, Binding] | None = None,
) -> list[list[Type]]:
    hints: dict[str, set[str]] = {name: set() for name in names}
    for child, parameter, constants in schema_nodes(node, row_parameter, bindings or {}):
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
                name = column_name(value, parameter, constants)
                name = names[name] if isinstance(name, int) else name
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
    choices = schema_types(candidate.node, names, row_parameter, candidate.bindings)
    initial = (0,) * len(names)
    pending: list[tuple[int, tuple[int, ...]]] = [(0, initial)]
    visited: set[tuple[int, ...]] = {initial}
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
