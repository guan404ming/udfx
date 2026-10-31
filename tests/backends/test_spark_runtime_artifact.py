import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def test_spark_runtime_matches_sources() -> None:
    directory = Path(__file__).resolve().parents[2] / "src/udfx/backends/spark/runtime"
    metadata = json.loads((directory / "runtime.json").read_text())
    digest = hashlib.sha256()
    sources = sorted(directory.glob("*.java"))
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    assert metadata["source_sha256"] == digest.hexdigest()
    assert metadata["package"] == "udfx.runtime.v" + digest.hexdigest()[:16]
    jar = directory / f"udfx-runtime-{metadata['source_sha256']}.jar"
    assert list(directory.glob("*.jar")) == [jar]
    assert metadata["jar_sha256"] == hashlib.sha256(jar.read_bytes()).hexdigest()
    with ZipFile(jar) as archive:
        assert set(archive.namelist()) == {
            metadata["package"].replace(".", "/") + "/" + source.stem + ".class"
            for source in sources
        }
        for name in archive.namelist():
            assert int.from_bytes(archive.read(name)[6:8]) == metadata["java_release"] + 44
