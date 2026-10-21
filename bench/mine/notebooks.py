from __future__ import annotations

import ast
import csv
import sys
from io import StringIO
from pathlib import Path
from tokenize import FSTRING_MIDDLE, STRING, TokenError, generate_tokens

from bench.mine.bindings import notebook_bindings, select_bindings
from bench.mine.call_sites import find_candidates
from bench.mine.definitions import safe_definition, strip_definition
from bench.mine.models import Candidate, Counts
from bench.mine.provenance import candidate_origins, notebook_origins


def strip_magics(source: str) -> str:
    if not any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
        return source
    literal_lines: set[int] = set()
    try:
        for token in generate_tokens(StringIO(source).readline):
            if token.type in {STRING, FSTRING_MIDDLE}:
                literal_lines.update(range(token.start[0] + 1, token.end[0] + 1))
    except (TokenError, IndentationError):
        pass
    lines = []
    shell_continuation = False
    for position, line in enumerate(source.splitlines(keepends=True), 1):
        if position in literal_lines:
            lines.append(line)
            continue
        stripped = line.lstrip()
        shell_line = shell_continuation or stripped.startswith("!")
        shell_continuation = shell_line and line.rstrip().endswith("\\")
        lines.append("\n" if shell_line or stripped.startswith("%") else line)
    return "".join(lines)


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
    counts = Counts(data_file=str(path))
    notebooks = read_notebooks(path, counts)
    unique: dict[
        tuple[str, str, tuple[tuple[str, str], ...], tuple[tuple[str, str | None], ...]], Candidate
    ] = {}
    for kernel_id, cells in notebooks.items():
        trees: list[ast.Module] = []
        references: list[tuple[int, str, str, ast.FunctionDef | ast.Lambda | ast.Name]] = []
        for position, source in cells:
            cleaned = strip_magics(source)
            counts.magic_cells += cleaned != source
            try:
                tree = ast.parse(cleaned)
            except (SyntaxError, ValueError, RecursionError):
                counts.skipped_cells["parse error"] += 1
                continue
            trees.append(tree)
            references.extend(
                (position, site, kind, node) for site, kind, node in find_candidates(tree)
            )
        available = notebook_bindings(trees)
        origins = notebook_origins(trees, available)
        for position, site, kind, reference in references:
            counts.call_sites[site] += 1
            node = available.get(reference.id) if isinstance(reference, ast.Name) else reference
            if not isinstance(node, ast.FunctionDef | ast.Lambda):
                assert isinstance(reference, ast.Name)
                counts.unresolved[reference.id] += 1
                continue
            stripped = strip_definition(node)
            if not safe_definition(stripped):
                counts.skipped_definitions["definition-time expressions remain"] += 1
                continue
            try:
                bindings = select_bindings(stripped, available)
                name_origins = candidate_origins(stripped, bindings, origins)
            except (SyntaxError, ValueError, TypeError, RecursionError) as error:
                counts.skipped_definitions[f"definition error: {type(error).__name__}"] += 1
                continue
            key = (
                kind,
                ast.dump(stripped),
                tuple((name, ast.dump(binding)) for name, binding in sorted(bindings.items())),
                tuple(name_origins.items()),
            )
            if key in unique:
                candidate = unique[key]
                candidate.occurrences += 1
                if position < candidate.first_cell:
                    candidate.first_cell = position
                    candidate.kernel_id = kernel_id
            else:
                unique[key] = Candidate(
                    kernel_id,
                    position,
                    stripped,
                    kind=kind,
                    bindings=bindings,
                    name_origins=name_origins,
                )
    return sorted(unique.values(), key=lambda candidate: candidate.first_cell), counts
