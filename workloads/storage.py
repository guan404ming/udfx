import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pyarrow
from pyarrow import parquet

DIRECTORY = Path(__file__).parent
FIXTURES = DIRECTORY / "fixtures"
DEFAULT_CACHE = Path(".cache/workloads")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_manifest(suite: str) -> dict:
    return json.loads((DIRECTORY / "sources" / f"{suite}.json").read_text())


def download(url: str, checksum: str, destination: Path) -> Path:
    if destination.exists():
        if digest(destination) != checksum:
            raise ValueError(f"Checksum mismatch: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if digest(temporary) != checksum:
            raise ValueError(f"Checksum mismatch: {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_tables(
    suite: str, tables: dict[str, pyarrow.Table], root: Path, *, row_limit: int | None = None
) -> None:
    target = root / suite
    target.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, table in tables.items():
        path = target / f"{name}.parquet"
        parquet.write_table(table.replace_schema_metadata(None), path)
        records[name] = {
            "sha256": digest(path),
            "rows": table.num_rows,
            "schema": str(table.schema.remove_metadata()),
        }
    metadata = {
        "schema_version": 1,
        "source": read_manifest(suite),
        "tables": records,
        "sampling": {"method": "source prefix", "row_limit": row_limit},
    }
    (target / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def verify_tables(suite: str, root: Path) -> dict:
    metadata = json.loads((root / suite / "manifest.json").read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported prepared-data schema: {suite}; prepare the data again")
    if metadata["source"] != read_manifest(suite):
        raise ValueError(f"Source manifest changed: {suite}; prepare the data again")
    for name, record in metadata["tables"].items():
        path = root / suite / f"{name}.parquet"
        if digest(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
    return metadata
