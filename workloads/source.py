import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from workloads.storage import digest, download, source_path

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
CHECKSUM = re.compile(r"[0-9a-fA-F]{64}")


def parse_sources(values: list[str]) -> dict[str, Path]:
    sources = {}
    for value in values:
        name, separator, directory = value.partition("=")
        if not separator or NAME.fullmatch(name) is None or not directory:
            raise ValueError(f"Expected NAME=DIR for a local source: {value}")
        if name in sources:
            raise ValueError(f"Duplicate local source: {name}")
        path = Path(directory).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Local source is not a directory: {path}")
        sources[name] = path
    return sources


def resolve_sources(
    manifest: dict, destination: Path, sources: dict[str, Path]
) -> tuple[list[Path], list[dict]]:
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("Source manifest needs a files list")
    targets = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Source manifest files must be records")
        target = source_path(destination, entry.get("path"))
        if target.resolve() in targets:
            raise ValueError(f"Duplicate source path: {entry['path']}")
        targets.add(target.resolve())

    def fetch(entry: dict) -> tuple[Path, dict]:
        origins = [name for name in ("url", "input") if name in entry]
        if len(origins) != 1:
            raise ValueError(f"Source needs exactly one url or input: {entry['path']}")
        origin = origins[0]
        value = entry[origin]
        checksum = entry.get("sha256")
        if checksum is not None and (
            not isinstance(checksum, str) or CHECKSUM.fullmatch(checksum) is None
        ):
            raise ValueError(f"Invalid SHA-256 checksum: {entry['path']}")
        checksum = checksum.lower() if checksum is not None else None
        if origin == "input":
            if not isinstance(value, str) or NAME.fullmatch(value) is None:
                raise ValueError(f"Invalid local source name: {value}")
            if value not in sources:
                raise ValueError(f"Missing local source: {value}")
            root = sources[value]
            if not root.is_dir():
                raise ValueError(f"Local source is not a directory: {root}")
            path = source_path(root, entry["path"])
            if not path.is_file():
                raise FileNotFoundError(f"Local source file not found: {path}")
            actual = digest(path)
            if checksum is not None and checksum != actual:
                raise ValueError(f"Checksum mismatch: {path}")
            checksum = actual
            url = path.resolve().as_uri()
        else:
            if not isinstance(value, str) or not value:
                raise ValueError(f"Source URL must be a nonempty string: {value}")
            if not checksum:
                raise ValueError(f"Source URL needs a SHA-256 checksum: {entry['path']}")
            url = value
        path = download(url, checksum, source_path(destination, entry["path"]))
        return path, {"path": entry["path"], "sha256": checksum, origin: value}

    with ThreadPoolExecutor(max_workers=4) as pool:
        resolved = list(pool.map(fetch, entries))
    return [path for path, _ in resolved], [record for _, record in resolved]
