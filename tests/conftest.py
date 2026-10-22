import os
from functools import cache
from pathlib import Path
from random import Random
from typing import Any

import pytest
from _pytest.mark import ParameterSet
from hypothesis import HealthCheck, settings
from pyspark.sql import SparkSession

from udfx.backends import installed
from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend

TESTS = str(Path(__file__).resolve().parent)
FULL = os.environ.get("UDFX_TESTS") == "full"
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [TESTS, os.environ.get("PYTHONPATH")]))
os.environ.setdefault(
    "PYSPARK_SUBMIT_ARGS",
    "--master local[1] --conf spark.ui.enabled=false --conf spark.sql.shuffle.partitions=1"
    " --conf spark.default.parallelism=1"
    " pyspark-shell",
)

settings.register_profile(
    "default", max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile("ci", max_examples=100, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(autouse=True)
def batch_seed(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    value = request.config.getoption("--hypothesis-seed")
    if value is None and settings().derandomize:
        value = request.node.nodeid
    if value is not None:
        random_seed = Random(value).getrandbits(64)
        monkeypatch.setattr("udfx.testing.batching.getrandbits", lambda _: random_seed)


@cache
def spark_session(ansi: bool) -> SparkSession:
    session = SparkSession.builder.getOrCreate().newSession()
    session.conf.set("spark.sql.ansi.enabled", str(ansi).lower())
    return session


class AnsiSparkBackend(SparkBackend):
    def __init__(self, ansi: bool) -> None:
        super().__init__()
        self.ansi = ansi

    def session(self) -> SparkSession:
        return spark_session(self.ansi)


def backends() -> list[tuple[str, Backend[Any]]]:
    found: list[tuple[str, Backend[Any]]] = []
    for backend in installed():
        if isinstance(backend, SparkBackend):
            found.append((backend.name, AnsiSparkBackend(ansi=True)))
            if FULL:
                found.append((f"{backend.name}-ansi-off", AnsiSparkBackend(ansi=False)))
        else:
            found.append((backend.name, backend))
    return sorted(found, key=lambda item: not isinstance(item[1], SparkBackend))


def backend_params() -> list[ParameterSet]:
    return [pytest.param(backend, id=label) for label, backend in backends()]


def frontend_backend_params() -> list[ParameterSet]:
    return [param for param in backend_params() if FULL or param.id == "polars"]


def engine_names() -> list[str]:
    return [backend.name for backend in installed() if FULL or backend.name == "polars"]
