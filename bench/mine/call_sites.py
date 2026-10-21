from __future__ import annotations

import ast

METHODS = {"apply", "map", "applymap", "transform", "agg", "aggregate", "map_elements", "map_rows"}
WRAPPERS = {"udf", "pandas_udf"}
GROUP_METHODS = {"groupby", "rolling", "expanding", "resample", "ewm"}


def call_site(function: ast.expr) -> str | None:
    if isinstance(function, ast.Name) and function.id in WRAPPERS:
        return function.id
    if isinstance(function, ast.Attribute):
        if function.attr in METHODS:
            return f".{function.attr}"
        if function.attr in WRAPPERS:
            return function.attr
        if (
            function.attr == "vectorize"
            and isinstance(function.value, ast.Name)
            and function.value.id == "np"
        ):
            return "np.vectorize"
    return None


def call_kind(function: ast.expr, keywords: list[ast.keyword]) -> str:
    site = call_site(function)
    if site in {".transform", ".agg", ".aggregate", "pandas_udf"}:
        return "group"
    receiver = function.value if isinstance(function, ast.Attribute) else None
    while receiver is not None:
        if isinstance(receiver, ast.Call):
            if isinstance(receiver.func, ast.Attribute) and receiver.func.attr in GROUP_METHODS:
                return "group"
            receiver = receiver.func
        elif isinstance(receiver, ast.Attribute | ast.Subscript):
            receiver = receiver.value
        else:
            break
    if site == ".map_rows" or (
        site == ".apply"
        and any(
            keyword.arg == "axis"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value in (1, "columns")
            for keyword in keywords
        )
    ):
        return "row"
    return "element"


def find_candidates(
    tree: ast.Module,
) -> list[tuple[str, str, ast.FunctionDef | ast.Lambda | ast.Name]]:
    found: list[tuple[str, str, ast.FunctionDef | ast.Lambda | ast.Name]] = []
    decorators: set[ast.expr] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                function = decorator.func if isinstance(decorator, ast.Call) else decorator
                site = call_site(function)
                if site in WRAPPERS:
                    found.append((f"@{site}", call_kind(function, []), node))
                    decorators.add(decorator)
        elif isinstance(node, ast.Call) and node not in decorators:
            site = call_site(node.func)
            if site is None:
                continue
            arguments = [
                *node.args[:1],
                *(
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in {"func", "f", "pyfunc", "function"}
                ),
            ]
            found.extend(
                (site, call_kind(node.func, node.keywords), argument)
                for argument in arguments
                if isinstance(argument, ast.Lambda | ast.Name)
            )
    return found
