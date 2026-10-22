from __future__ import annotations

from collections import Counter

from bench.mine.bindings import HELPER_DEPTH
from bench.mine.measurement import normalize_reason
from bench.mine.models import Counts, Result


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


def fraction(count: int, total: int) -> str:
    return f"{count:,}/{total:,} ({100 * count / total if total else 0:.2f}%)"


def coverage(results: list[Result]) -> tuple[str, str]:
    return (
        fraction(sum(result.ok for result in results), len(results)),
        fraction(
            sum(result.occurrences for result in results if result.ok),
            sum(result.occurrences for result in results),
        ),
    )


def failure_table(results: list[Result], kind: str) -> list[str]:
    categories: Counter[str] = Counter()
    weighted: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for result in results:
        if result.kind == kind and result.category is not None:
            categories[result.category] += 1
            weighted[result.category] += result.occurrences
            examples.setdefault(result.category, result.source)
    return [
        f"## Top 25 failure categories: {kind}",
        "",
        "| Category | Unique | Occurrences | Example source |",
        "| --- | ---: | ---: | --- |",
        *[
            f"| {table_text(category)} | {count:,} | {weighted[category]:,} | "
            f"{table_text(examples[category], 120)} |"
            for category, count in categories.most_common(25)
        ],
        "",
    ]


def summary(results: list[Result], counts: Counts) -> str:
    headline = [result for result in results if result.kind in {"element", "row"}]
    unique, weighted = coverage(headline)
    in_scope = [result for result in headline if not result.third_party_names]
    scope_unique, scope_weighted = coverage(in_scope)
    external = [result for result in headline if result.third_party_names]
    names: Counter[str] = Counter()
    weighted_names: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    weighted_missing: Counter[str] = Counter()
    for result in results:
        for name in result.third_party_names:
            names[name] += 1
            weighted_names[name] += result.occurrences
        if result.reason is None:
            continue
        reason = normalize_reason(result.reason)
        if reason.startswith(("no op ", "unknown callable ")):
            target = reason.split("(", 1)[0] if reason.startswith("no op ") else reason
            missing[target] += 1
            weighted_missing[target] += result.occurrences
    groups = {
        "element + row": headline,
        **{
            kind: [result for result in results if result.kind == kind]
            for kind in ("element", "row", "group")
        },
    }
    lines = [
        "# UDF compilation coverage",
        "",
        "## Headline: element + row",
        "",
        "### All candidates",
        "",
        f"- Compiled unique: {unique}",
        f"- Compiled weighted by occurrences: {weighted}",
        "",
        "### In-scope candidates",
        "",
        f"- Compiled unique: {scope_unique}",
        f"- Compiled weighted by occurrences: {scope_weighted}",
        "",
        "## Setup",
        "",
        f"- Data file: {table_text(counts.data_file)}",
        f"- Notebooks: {counts.notebooks:,}",
        f"- Cells: {counts.cells:,}",
        f"- Cells with magic or shell lines removed: {counts.magic_cells:,}",
        f"- Candidate references: {counts.call_sites.total():,}",
        "- Candidates total (resolved, safe occurrences): "
        f"{sum(result.occurrences for result in results):,}",
        f"- Candidates unique (source, globals, name origins, kind): {len(results):,}",
        f"- Collection seconds: {counts.collection_seconds:.2f}",
        f"- Compilation seconds: {counts.compile_seconds:.2f}",
        f"- CSV stop: {table_text(counts.csv_error) if counts.csv_error else 'end of file'}",
        "",
        "## Coverage per kind",
        "",
        "Group candidates are out of scope for scalar UDFs. "
        "Their compilation rate is diagnostic only.",
        "",
        "| Kind | Compiled unique | Compiled occurrences |",
        "| --- | ---: | ---: |",
        *[f"| {kind} | {' | '.join(coverage(group))} |" for kind, group in groups.items()],
        "",
        "## Exclusions",
        "",
        f"- Unresolved names at call sites: {counts.unresolved.total():,} occurrences; "
        f"{len(counts.unresolved):,} distinct names",
        f"- Unsafe or unbindable definitions: {counts.skipped_definitions.total():,} occurrences",
        f"- Parse errors: {counts.skipped_cells['parse error']:,} cells",
        "- All-candidate coverage keeps unresolved globals and missing modules as failures. "
        "In-scope coverage excludes candidates with third-party names.",
        "",
        "## Third-party names",
        "",
        f"- Excluded element + row candidates: {len(external):,} unique; "
        f"{sum(result.occurrences for result in external):,} occurrences",
        "- Name counts below include all kinds.",
        "",
        "| Name | Unique | Occurrences |",
        "| --- | ---: | ---: |",
        *[
            f"| {table_text(name)} | {count:,} | {weighted_names[name]:,} |"
            for name, count in names.most_common()
        ],
        "",
        "## Candidates by call site",
        "",
        "| Call site | References |",
        "| --- | ---: |",
        *[f"| {site} | {count:,} |" for site, count in counts.call_sites.most_common()],
        "",
        *[line for kind in ("element", "row", "group") for line in failure_table(results, kind)],
        "## Top 25 missing ops and callable targets across all kinds",
        "",
        "| Missing op or callable | Unique | Occurrences |",
        "| --- | ---: | ---: |",
        *[
            f"| {table_text(target)} | {count:,} | {weighted_missing[target]:,} |"
            for target, count in missing.most_common(25)
        ],
        "",
        "## Method and assumptions",
        "",
        "1. Parse cells with ast. Never execute notebook code or call candidates or helpers. "
        "Bind only through bind_function. Strip decorators, defaults, and annotations, including "
        "nested definitions. Reject remaining definition-time expressions. These removals can "
        "change the original function contract.",
        "2. Use CSV order within kernel_id as cell order. Ignore cell_index and execution_count. "
        "Resolve across the entire notebook, including later cells. Prefer the last module-level "
        "def, then the last simple or annotated assignment, then an import. Evaluate assignments "
        "only with ast.literal_eval. Leave an invalid last assignment missing. Ignore conditional "
        "definitions, destructuring, aliases, and star imports.",
        "3. Read global names from bytecode, including nested code. Resolve helper dependencies "
        f"through depth {HELPER_DEPTH}, with direct helpers at depth zero. Stop cycles. Share a "
        "globals dictionary between helpers. Keep standard builtins. Import only math, re, "
        "string, datetime, unicodedata, numpy, pandas, os, and os.path by their real names with "
        "importlib. Reject relative imports and other submodules. "
        "Resolve from-imports from module dictionaries. "
        "Leave unresolved names missing. Report missing installed modules as failures.",
        "4. Recognize inline or assigned lambdas, module-level defs, "
        "and udf/pandas_udf decorators. "
        "Use the first positional argument or a func/f/pyfunc/function keyword. Classify by method "
        "spelling without receiver type inference. "
        "A visible groupby/rolling/expanding/resample/ewm "
        "receiver chain overrides row and element. transform/agg/aggregate and pandas_udf are "
        "group. map_rows and apply with literal axis=1 or axis='columns' are row. Other recognized "
        "sites are element. Do not infer receiver aliases or dynamic/positional axes.",
        "5. Deduplicate stripped source AST plus dependency ASTs, relevant name origins, and kind, "
        "without locations. Count once per kind and global context. Weight by reference "
        "occurrences, not runtime calls or dataframe rows. Exclude unresolved call-site names "
        "and unsafe definitions. Remove magic and shell lines, including shell line continuations. "
        "Parse the rest of each cell. Skip parse-error cells. Stop at malformed CSV rows.",
        "6. Compile with the default Polars backend. Search nullable STR, FLOAT, INT, and BOOL "
        "per parameter, up to 64 schemas. String methods and string operands prefer STR; "
        "arithmetic prefers FLOAT then INT; boolean operations prefer BOOL. String hints take "
        "priority over arithmetic, then boolean hints. Without hints, use STR, FLOAT, INT, BOOL. "
        "Visit combinations by total preference rank, then index tuple. Save the first success. "
        "Use the first attempted failure when every schema fails.",
        "7. Require one parameter for rows. Infer columns from constant string or integer "
        "subscripts and attributes on that parameter. Resolve local keys with one assignment "
        "in the same function. Follow row arguments into resolved helpers through the same "
        "depth limit. Keep column type hints from those helpers. Use the same schema search. "
        "Reject absent columns or columns identical to the parameter list. "
        "Pass inferred schemas to udfx. Keep unsupported row schemas as explicit failures.",
        "8. Rank failures and missing targets by unique count. Missing overloads can reflect "
        "schema guesses. Resolved helpers can remain unsupported callables. Compilation success "
        "measures acceptance only. It does not validate input types, runtime results, other "
        "backends, or group semantics.",
        "9. Mark a candidate as third-party only when its recorded failure is unknown global X "
        "or missing module X and X resolves through a notebook import to a module outside the "
        "exact allowlist, or X has no resolvable import and no definition in the notebook. "
        "Track module-scope declarations, including conditional and destructured bindings, "
        "for this check without executing them. Missing allowlisted modules stay in scope. "
        "Store the failure name in third_party_names; use an empty list for other candidates. "
        "Report element + row coverage for all candidates and for in-scope candidates, "
        "excluding third-party candidates from both unique and occurrence denominators.",
        "",
    ]
    return "\n".join(lines)
