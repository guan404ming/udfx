import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import pyarrow
from pyarrow import compute, csv, parquet

from workloads.storage import DEFAULT_CACHE, download, read_manifest, save_tables

FLIGHT_COLUMNS = {
    "FlightDate": "date",
    "Reporting_Airline": "carrier",
    "Origin": "origin",
    "OriginCityName": "city",
    "Dest": "destination",
    "DepTime": "departure_time",
    "ArrDelay": "arrival_delay",
    "Cancelled": "cancelled",
    "Distance": "distance",
}


def flights(paths: list[Path]) -> dict[str, pyarrow.Table]:
    types = {name: pyarrow.string() for name in FLIGHT_COLUMNS}
    with ZipFile(paths[0]) as archive:
        filename = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(filename) as stream:
            table = csv.read_csv(
                stream,
                convert_options=csv.ConvertOptions(
                    include_columns=list(types),
                    column_types=types,
                    strings_can_be_null=True,
                ),
            ).rename_columns(list(FLIGHT_COLUMNS.values()))
    for name, type in (
        ("departure_time", pyarrow.int64()),
        ("arrival_delay", pyarrow.float64()),
        ("cancelled", pyarrow.float64()),
        ("distance", pyarrow.float64()),
    ):
        table = table.set_column(table.schema.get_field_index(name), name, table[name].cast(type))
    airports = (
        table.select(["origin", "city"])
        .group_by(["origin", "city"])
        .aggregate([])
        .sort_by([("origin", "ascending"), ("city", "ascending")])
    )
    if len(set(airports["origin"].to_pylist())) != airports.num_rows:
        raise ValueError("The airport lookup needs one city per airport")
    return {"flights": table, "airports": airports.rename_columns(["airport", "city"])}


def nyc311(paths: list[Path]) -> dict[str, pyarrow.Table]:
    names = [
        "unique_key",
        "created_date",
        "closed_date",
        "agency",
        "complaint_type",
        "descriptor",
        "incident_zip",
        "borough",
    ]
    table = csv.read_csv(
        paths[0],
        convert_options=csv.ConvertOptions(
            column_types=dict.fromkeys(names, pyarrow.string()),
            strings_can_be_null=True,
        ),
    )
    for name in ("created_date", "closed_date"):
        table = table.set_column(
            table.schema.get_field_index(name),
            name,
            compute.cast(table[name], pyarrow.timestamp("us")),
        )
    return {"requests": table}


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
    "flights": flights,
    "nyc311": nyc311,
}


def prepare(suite: str, root: Path, rows: int | None = None) -> None:
    manifest = read_manifest(suite)
    entries = manifest["files"]

    def fetch(entry: dict) -> Path:
        return download(entry["url"], entry["sha256"], root / suite / "source" / entry["path"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch, entries))
    tables = PREPARERS[suite](paths)
    if rows is not None:
        full_tables = manifest.get("dimension_tables", [])
        tables = {
            name: table if name in full_tables else table.slice(0, rows)
            for name, table in tables.items()
        }
    save_tables(suite, tables, root, row_limit=rows)


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and verify pinned public benchmark data")
    parser.add_argument("--suites", nargs="+", choices=PREPARERS, default=list(PREPARERS))
    parser.add_argument("--root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rows", type=int)
    options = parser.parse_args(arguments)
    if options.rows is not None and options.rows <= 0:
        parser.error("--rows must be positive")
    for suite in options.suites:
        prepare(suite, options.root, options.rows)
        print(f"Prepared {suite}: {options.root / suite}")


if __name__ == "__main__":
    main()
