import argparse
from collections.abc import Callable
from pathlib import Path

import pyarrow
from pyarrow import parquet

from workloads import qure
from workloads.catalog import DEFAULT_SUITES
from workloads.source import parse_sources, resolve_sources
from workloads.storage import DEFAULT_CACHE, read_manifest, save_tables


def udfbench(paths: list[Path]) -> dict[str, pyarrow.Table]:
    tables = {path.stem: parquet.read_table(path) for path in paths if path.suffix == ".parquet"}
    authors = tables["artifact_authorlists"]
    shifted = pyarrow.concat_arrays(
        [authors["1"].slice(1).combine_chunks(), authors["1"].slice(0, 1).combine_chunks()]
    )
    tables["artifact_authorlists"] = authors.append_column("next_authorlist", shifted)
    return tables


PREPARERS: dict[str, Callable[[list[Path]], dict[str, pyarrow.Table]]] = {
    "udfbench": udfbench,
    "qure": qure.prepare,
}


def prepare(
    suite: str,
    root: Path,
    rows: int | None = None,
    sources: dict[str, Path] | None = None,
    selected: set[str] | None = None,
) -> None:
    manifest = read_manifest(suite)
    paths, records = resolve_sources(manifest, root / suite / "source", sources or {})
    tables = PREPARERS[suite](paths) if selected is None else qure.prepare(paths, selected)
    if rows is not None:
        full_tables = manifest.get("dimension_tables", [])
        tables = {
            name: table if name in full_tables else table.slice(0, rows)
            for name, table in tables.items()
        }
    save_tables(suite, tables, root, row_limit=rows, sources=records)


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare and verify benchmark sources")
    parser.add_argument("--suites", nargs="+", choices=PREPARERS, default=list(DEFAULT_SUITES))
    parser.add_argument("--root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--source", action="append", default=[], metavar="NAME=DIRECTORY")
    parser.add_argument("--cases", nargs="+")
    options = parser.parse_args(arguments)
    if options.rows is not None and options.rows <= 0:
        parser.error("--rows must be positive")
    if len(options.suites) != len(set(options.suites)):
        parser.error("--suites must contain unique names")
    if options.cases is not None and options.suites != ["qure"]:
        parser.error("--cases requires --suites qure")
    try:
        selected = None if options.cases is None else qure.case_tables(set(options.cases))
        sources = parse_sources(options.source)
        required = {
            entry["input"]
            for suite in options.suites
            for entry in read_manifest(suite)["files"]
            if "input" in entry
        }
        if sources.keys() - required:
            raise ValueError(
                f"Unknown source names: {', '.join(sorted(sources.keys() - required))}"
            )
        if required - sources.keys():
            raise ValueError(f"Missing sources: {', '.join(sorted(required - sources.keys()))}")
    except ValueError as error:
        parser.error(str(error))
    for suite in options.suites:
        prepare(suite, options.root, options.rows, sources, selected)
        print(f"Prepared {suite}: {options.root / suite}")


if __name__ == "__main__":
    main()
