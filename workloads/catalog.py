from collections.abc import Callable
from pathlib import Path

from workloads import flights, nyc311, qure, udfbench
from workloads.models import Workload

DEFAULT_SUITES = ("udfbench", "flights", "nyc311")

SUITES: dict[str, Callable[..., list[Workload]]] = {
    "udfbench": udfbench.load,
    "flights": flights.load,
    "nyc311": nyc311.load,
    "qure": qure.load,
}


def load(
    suites: list[str],
    root: Path,
    cases: set[str] | None = None,
    *,
    unavailable: set[str] | None = None,
) -> list[Workload]:
    workloads = [
        workload
        for suite in suites
        for workload in SUITES[suite](
            root, cases, load_functions=suite not in (unavailable or set())
        )
    ]
    identifiers = [workload.identifier for workload in workloads]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Workload identifiers must be unique")
    if cases is not None and (unknown := cases - set(identifiers)):
        raise ValueError(f"Unknown workload identifiers: {', '.join(sorted(unknown))}")
    return workloads
