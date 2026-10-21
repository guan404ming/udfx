from __future__ import annotations

import argparse
import ast
import builtins
import csv
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import FunctionType

import udfx
from udfx.types import FLOAT, INT, STR

METHODS = {"apply", "map", "applymap", "transform", "agg", "map_elements", "map_rows"}
WRAPPERS = {"udf", "pandas_udf"}


@dataclass
class Counts:
    notebooks: int = 0
    cells: int = 0
    skipped_cells: Counter[str] = field(default_factory=Counter)
    call_sites: Counter[str] = field(default_factory=Counter)
    unresolved: Counter[str] = field(default_factory=Counter)
    skipped_definitions: Counter[str] = field(default_factory=Counter)
    csv_error: str | None = None


@dataclass
class Candidate:
    kernel_id: str
    first_cell: int
    node: ast.FunctionDef | ast.Lambda
    occurrences: int = 1


@dataclass
class Result:
    kernel_id: str
    occurrences: int
    source: str
    ok: bool = False
    schema: dict[str, str] | None = None
    reason: str | None = None
    category: str | None = None


def strip_definition(node: ast.FunctionDef | ast.Lambda) -> ast.FunctionDef | ast.Lambda:
    stripped = deepcopy(node)
    for child in ast.walk(stripped):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            child.decorator_list = []
            child.returns = None
            child.type_comment = None
        elif isinstance(child, ast.arguments):
            child.defaults = []
            child.kw_defaults = [None] * len(child.kwonlyargs)
        elif isinstance(child, ast.arg):
            child.annotation = None
            child.type_comment = None
    return stripped


def safe_definition(node: ast.FunctionDef | ast.Lambda) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and (
            child.decorator_list or child.returns is not None or child.type_params
        ):
            return False
        if isinstance(child, ast.arguments) and (
            child.defaults or any(value is not None for value in child.kw_defaults)
        ):
            return False
        if isinstance(child, ast.arg) and child.annotation is not None:
            return False
    return True


def bind_function(node: ast.FunctionDef | ast.Lambda) -> FunctionType:
    stripped = strip_definition(node)
    if not safe_definition(stripped):
        raise ValueError("definition-time expressions remain")
    name = stripped.name if isinstance(stripped, ast.FunctionDef) else "udfx_candidate"
    statement = (
        stripped
        if isinstance(stripped, ast.FunctionDef)
        else ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=stripped)
    )
    module = ast.fix_missing_locations(ast.Module(body=[statement], type_ignores=[]))
    namespace: dict[str, object] = {"__builtins__": builtins}
    exec(builtins.compile(module, "<udfx-candidate>", "exec", dont_inherit=True), namespace)
    function = namespace[name]
    assert isinstance(function, FunctionType)
    return function


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


def find_candidates(tree: ast.Module) -> list[tuple[str, ast.FunctionDef | ast.Lambda | ast.Name]]:
    found: list[tuple[str, ast.FunctionDef | ast.Lambda | ast.Name]] = []
    decorators: set[ast.expr] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                function = decorator.func if isinstance(decorator, ast.Call) else decorator
                site = call_site(function)
                if site in WRAPPERS:
                    found.append((f"@{site}", node))
                    decorators.add(decorator)
        elif isinstance(node, ast.Call) and node not in decorators:
            site = call_site(node.func)
            if site is None:
                continue
            arguments = (
                node.args[:1]
                if site.startswith(".")
                else [*node.args, *(keyword.value for keyword in node.keywords)]
            )
            found.extend(
                (site, argument)
                for argument in arguments
                if isinstance(argument, ast.Lambda | ast.Name)
            )
    return found


def read_notebooks(path: Path, counts: Counts) -> dict[str, list[tuple[int, str]]]:
    csv.field_size_limit(sys.maxsize)
    notebooks: dict[str, list[tuple[int, str]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, strict=True)
        try:
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    counts.csv_error = f"incomplete row at line {reader.line_num}"
                    break
                counts.cells += 1
                notebooks.setdefault(row["kernel_id"], []).append((counts.cells, row["source"]))
        except csv.Error as error:
            counts.csv_error = f"{error} at line {reader.line_num}"
    counts.notebooks = len(notebooks)
    return notebooks


def collect_candidates(path: Path) -> tuple[list[Candidate], Counts]:
    counts = Counts()
    notebooks = read_notebooks(path, counts)
    unique: dict[str, Candidate] = {}
    for kernel_id, cells in notebooks.items():
        definitions: dict[str, ast.FunctionDef] = {}
        references: list[tuple[int, str, ast.FunctionDef | ast.Lambda | ast.Name]] = []
        for position, source in cells:
            if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
                counts.skipped_cells["IPython magic or shell line"] += 1
                continue
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError, RecursionError):
                counts.skipped_cells["parse error"] += 1
                continue
            definitions.update(
                (node.name, node) for node in tree.body if isinstance(node, ast.FunctionDef)
            )
            references.extend((position, site, node) for site, node in find_candidates(tree))
        for position, site, reference in references:
            counts.call_sites[site] += 1
            node = definitions.get(reference.id) if isinstance(reference, ast.Name) else reference
            if node is None:
                assert isinstance(reference, ast.Name)
                counts.unresolved[reference.id] += 1
                continue
            stripped = strip_definition(node)
            if not safe_definition(stripped):
                counts.skipped_definitions["definition-time expressions remain"] += 1
                continue
            key = ast.dump(stripped)
            if key in unique:
                candidate = unique[key]
                candidate.occurrences += 1
                if position < candidate.first_cell:
                    candidate.first_cell = position
                    candidate.kernel_id = kernel_id
            else:
                unique[key] = Candidate(kernel_id, position, stripped)
    return sorted(unique.values(), key=lambda candidate: candidate.first_cell), counts


def normalize_reason(reason: str) -> str:
    reason = re.sub(r" at [\w.<>]+\+\d+$", "", reason)
    return re.sub(r"0x[0-9a-fA-F]+", "<address>", reason)


def failure_category(reason: str) -> str:
    reason = normalize_reason(reason)
    for prefix in ("unknown callable ", "unknown global "):
        if reason.startswith(prefix):
            return prefix.rstrip()
    if reason.startswith("no op "):
        return reason.split("(", 1)[0]
    return reason


def measure(candidate: Candidate) -> Result:
    result = Result(candidate.kernel_id, candidate.occurrences, ast.unparse(candidate.node))
    try:
        function = bind_function(candidate.node)
    except (SyntaxError, ValueError, TypeError, RecursionError) as error:
        result.reason = f"definition error: {type(error).__name__}: {error}"
        result.category = f"definition error: {type(error).__name__}"
        return result
    arguments = candidate.node.args
    parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    parameters.extend(argument for argument in (arguments.vararg, arguments.kwarg) if argument)
    for column_type in (STR.or_none(), FLOAT.or_none(), INT.or_none()):
        schema = {parameter.arg: column_type for parameter in parameters}
        try:
            translation = udfx.compile(function, schema)
        except Exception as error:
            reason = f"compiler error: {type(error).__name__}: {error}"
        else:
            if translation.ok:
                result.ok = True
                result.schema = {name: str(value) for name, value in schema.items()}
                result.reason = None
                result.category = None
                return result
            assert translation.unsupported is not None
            reason = str(translation.unsupported)
        if result.reason is None:
            result.reason = reason
            result.category = failure_category(reason)
    return result


def table_text(value: str, limit: int | None = None) -> str:
    value = " ".join(value.split())
    if limit is not None and len(value) > limit:
        value = value[: limit - 3] + "..."
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "&#124;")
        .replace("`", "&#96;")
    )


def summary(results: list[Result], counts: Counts) -> str:
    occurrences = sum(result.occurrences for result in results)
    successes = sum(result.ok for result in results)
    weighted = sum(result.occurrences for result in results if result.ok)
    categories: Counter[str] = Counter()
    weighted_categories: Counter[str] = Counter()
    examples: dict[str, str] = {}
    missing: Counter[str] = Counter()
    weighted_missing: Counter[str] = Counter()
    for result in results:
        if result.category is None or result.reason is None:
            continue
        categories[result.category] += 1
        weighted_categories[result.category] += result.occurrences
        examples.setdefault(result.category, result.source)
        reason = normalize_reason(result.reason)
        if reason.startswith(("no op ", "unknown callable ")):
            target = reason.split("(", 1)[0] if reason.startswith("no op ") else reason
            missing[target] += 1
            weighted_missing[target] += result.occurrences
    lines = [
        "# UDF compilation pilot",
        "",
        f"- Notebooks: {counts.notebooks:,}",
        f"- Cells: {counts.cells:,}",
        f"- Candidate references: {counts.call_sites.total():,}",
        f"- Candidates total (resolved, safe occurrences): {occurrences:,}",
        f"- Candidates unique: {len(results):,}",
        f"- Unresolved names: {counts.unresolved.total():,} occurrences; "
        f"{len(counts.unresolved):,} distinct names",
        f"- Skipped definitions: {counts.skipped_definitions.total():,}",
        f"- Compiled unique: {successes:,}/{len(results):,} "
        f"({100 * successes / len(results) if results else 0:.2f}%)",
        f"- Compiled weighted by occurrences: {weighted:,}/{occurrences:,} "
        f"({100 * weighted / occurrences if occurrences else 0:.2f}%)",
        *[f"- Skipped cells, {name}: {count:,}" for name, count in counts.skipped_cells.items()],
        f"- CSV stop: {table_text(counts.csv_error) if counts.csv_error else 'end of file'}",
        "",
        "Compilation uses the default Polars backend and tries all-nullable STR, FLOAT, then INT. "
        "Every parameter has the same type in each attempt. No candidate function is called.",
        "",
        "Names resolve to the last module-level function definition in CSV order within the "
        "same notebook. Globals and imports are not restored. Defaults, annotations, and "
        "decorators are removed. Cells with lines starting with % or ! are skipped. "
        "Unresolved names and unsafe definitions are excluded from compilation percentages.",
        "",
        "Failures use the first STR attempt. Counts rank unique candidates; occurrences "
        "show duplicate frequency. A missing overload can reflect the assumed schema. "
        "Compilation success does not validate real input types or runtime results.",
        "",
        "## Candidates by call site",
        "",
        "| Call site | References |",
        "| --- | ---: |",
        *[f"| {site} | {count:,} |" for site, count in counts.call_sites.most_common()],
        "",
        "## Top 25 failure categories",
        "",
        "| Category | Unique | Occurrences | Example source |",
        "| --- | ---: | ---: | --- |",
        *[
            f"| {table_text(category)} | {count:,} | {weighted_categories[category]:,} | "
            f"{table_text(examples[category], 120)} |"
            for category, count in categories.most_common(25)
        ],
        "",
        "## Top 15 missing ops and callable targets",
        "",
        "| Missing op or callable | Unique | Occurrences |",
        "| --- | ---: | ---: |",
        *[
            f"| {table_text(target)} | {count:,} | {weighted_missing[target]:,} |"
            for target, count in missing.most_common(15)
        ],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    candidates, counts = collect_candidates(arguments.csv)
    print(f"Read {counts.cells:,} cells; measure {len(candidates):,} unique candidates", flush=True)
    arguments.output.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    with (arguments.output / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for index, candidate in enumerate(candidates, 1):
            result = measure(candidate)
            results.append(result)
            stream.write(json.dumps(asdict(result), ensure_ascii=True) + "\n")
            if index % 1000 == 0:
                print(f"Measured {index:,}/{len(candidates):,}", flush=True)
    (arguments.output / "summary.md").write_text(summary(results, counts), encoding="utf-8")
    print(arguments.output / "summary.md")


if __name__ == "__main__":
    main()
