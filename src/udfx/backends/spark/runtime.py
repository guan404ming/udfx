import hashlib
import json
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from threading import RLock
from weakref import WeakKeyDictionary

from pyspark.sql import Column, SparkSession, functions

from udfx.backends.lowering_context import record_runtime

KERNELS = {
    "frequency_order": ("FrequencyOrder", "array<string>"),
    "json_words": ("JsonWords", "string"),
    "json_strings": ("JsonStrings", "string"),
}
CURRENT: ContextVar[Callable[[], SparkSession] | None] = ContextVar(
    "udfx spark runtime", default=None
)
LOCK = RLock()


@dataclass
class Registration:
    loaded: bool = False
    unavailable: bool = False
    functions: dict[str, str] = field(default_factory=dict)


REGISTRATIONS: WeakKeyDictionary[SparkSession, Registration] = WeakKeyDictionary()


@cache
def artifact() -> tuple[Path, str]:
    directory = Path(__file__).with_suffix("")
    metadata = json.loads((directory / "runtime.json").read_text())
    path = directory / f"udfx-runtime-{metadata['source_sha256']}.jar"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["jar_sha256"]:
        raise ValueError("Spark runtime artifact checksum mismatch")
    return path, str(metadata["package"])


@contextmanager
def scope(session: Callable[[], SparkSession]) -> Generator[None]:
    if CURRENT.get() == session:
        yield
        return
    token = CURRENT.set(session)
    try:
        yield
    finally:
        CURRENT.reset(token)


def registered(session: SparkSession, name: str) -> str | None:
    with LOCK:
        registration = REGISTRATIONS.setdefault(session, Registration())
        if registration.unavailable:
            return None
        if name in registration.functions:
            return registration.functions[name]
        class_name, result = KERNELS[name]
        try:
            path, package = artifact()
            if not registration.loaded:
                session.sql(f"ADD JAR '{path.as_uri()}'")
                registration.loaded = True
            function = f"udfx runtime {package.rsplit('.', 1)[-1]} {name}"
            session.udf.registerJavaFunction(function, f"{package}.{class_name}", result)
        except Exception as error:
            registration.unavailable = True
            warnings.warn(
                "udfx Spark runtime is unavailable; using engine expressions: "
                + (str(error).partition("\n")[0] or type(error).__name__),
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        registration.functions[name] = function
        return function


def call(name: str, *arguments: Column) -> Column | None:
    session = CURRENT.get()
    if session is None:
        return None
    function = registered(session(), name)
    if function is None:
        return None
    expr = functions.call_udf(f"`{function}`", *arguments)
    record_runtime(f"spark.{name}")
    return expr
