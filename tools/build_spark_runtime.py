import argparse
import hashlib
import io
import json
import subprocess
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile, ZipInfo

DIRECTORY = Path(__file__).resolve().parents[1] / "src/udfx/backends/spark/runtime"
JAVA_RELEASE = 17


def sources() -> tuple[dict[str, str], str]:
    values = {path.name: path.read_text() for path in sorted(DIRECTORY.glob("*.java"))}
    digest = hashlib.sha256()
    for name, value in values.items():
        digest.update(name.encode())
        digest.update(value.encode())
    return values, digest.hexdigest()


def build() -> tuple[str, bytes, bytes]:
    values, source_digest = sources()
    package = "udfx.runtime.v" + source_digest[:16]
    module = find_spec("pyspark")
    if module is None or module.origin is None:
        raise RuntimeError("Install the Spark development dependencies before rebuilding")
    jars = Path(module.origin).parent / "jars"
    with TemporaryDirectory(prefix="udfx-runtime-") as temporary:
        directory = Path(temporary)
        classes = directory / "classes"
        classes.mkdir()
        paths = []
        for name, value in values.items():
            path = directory / name
            path.write_text(value.replace("package udfx.runtime;", f"package {package};"))
            paths.append(str(path))
        subprocess.run(
            [
                "javac",
                "-proc:none",
                "-g:none",
                "--release",
                str(JAVA_RELEASE),
                "-encoding",
                "UTF-8",
                "-cp",
                str(jars / "*"),
                "-d",
                str(classes),
                *paths,
            ],
            check=True,
        )
        output = io.BytesIO()
        with ZipFile(output, "w", compression=ZIP_STORED) as archive:
            for path in sorted(classes.rglob("*.class")):
                entry = ZipInfo(path.relative_to(classes).as_posix(), (1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, path.read_bytes())
    jar = output.getvalue()
    metadata = {
        "package": package,
        "source_sha256": source_digest,
        "jar_sha256": hashlib.sha256(jar).hexdigest(),
        "java_release": JAVA_RELEASE,
    }
    return source_digest, jar, (json.dumps(metadata, indent=2) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    source_digest, jar, metadata = build()
    filename = f"udfx-runtime-{source_digest}.jar"
    for name, value in ((filename, jar), ("runtime.json", metadata)):
        path = DIRECTORY / name
        if arguments.check:
            if not path.exists() or path.read_bytes() != value:
                raise SystemExit(f"Rebuild the Spark runtime: {path} is stale")
        else:
            path.write_bytes(value)
    if not arguments.check:
        for path in [DIRECTORY / "runtime.jar", *DIRECTORY.glob("udfx-runtime-*.jar")]:
            if path.name != filename:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
