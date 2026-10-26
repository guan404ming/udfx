import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from coverage import Coverage

SCHEMA_VERSION = 1


def source_digest() -> str:
    digest = hashlib.sha256()
    for root in ("src", "bench", "workloads"):
        for path in sorted(Path(root).rglob("*")):
            if path.suffix in {".py", ".json", ".parquet"}:
                digest.update(str(path).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def git(*arguments: str) -> str:
    result = subprocess.run(["git", *arguments], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def metadata() -> dict:
    processor = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        processor = next(
            (
                line.partition(":")[2].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            processor,
        )
    elif sys.platform == "darwin":
        processor = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return {
        "schema_version": SCHEMA_VERSION,
        "started_at": datetime.now(UTC).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": source_digest(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "logical_cpus": os.cpu_count(),
        "load_average": list(os.getloadavg()),
        "versions": {
            name: version(name)
            for name in ("udfx", "pyarrow", "polars", "duckdb", "datafusion", "pyspark")
        },
        "instrumented": Coverage.current() is not None or sys.gettrace() is not None,
    }


def write(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported result schema: {artifact.get('schema_version')}")
    return artifact
