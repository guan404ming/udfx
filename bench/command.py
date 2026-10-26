import argparse
import fcntl
import os
from dataclasses import asdict
from pathlib import Path

from bench.artifacts import metadata, write
from bench.runner import ENGINES, MODES, record, run
from bench.timing import Settings
from workloads.catalog import SUITES, load
from workloads.storage import DEFAULT_CACHE, verify_tables


def parse_arguments(
    arguments: list[str] | None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Validate and benchmark Python UDFs on pinned public data"
    )
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=Path(".cache/results/benchmark.json"))
    parser.add_argument("--rows", type=int)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    options = parser.parse_args(arguments)
    for name in ("suites", "engines"):
        values = getattr(options, name)
        if len(values) != len(set(values)):
            parser.error(f"--{name} must contain unique names")
    return parser, options


def main(arguments: list[str] | None = None) -> int:
    parser, options = parse_arguments(arguments)
    try:
        settings = Settings(options.warmups, options.rounds, options.threads)
        if options.rows is not None and options.rows < 1:
            raise ValueError("--rows must be positive")
    except ValueError as error:
        parser.error(str(error))
    os.environ["POLARS_MAX_THREADS"] = str(settings.threads)
    workloads = load(options.suites, options.root)
    if options.cases:
        selected = set(options.cases)
        unknown = selected - {workload.identifier for workload in workloads}
        if unknown:
            parser.error(f"Unknown workload identifiers: {', '.join(sorted(unknown))}")
        workloads = [workload for workload in workloads if workload.identifier in selected]
    artifact: dict = {
        **metadata(),
        "settings": asdict(settings),
        "datasets": {},
        "complete": False,
        "results": [
            record(workload, engine, mode, "pending")
            for workload in workloads
            for engine in options.engines
            for mode in MODES
        ],
    }
    if artifact["instrumented"]:
        parser.error("Run benchmarks without coverage or a debugger")
    errors = {}
    for suite in options.suites:
        try:
            artifact["datasets"][suite] = verify_tables(suite, options.root)
        except Exception as error:
            errors[suite] = f"{type(error).__name__}: {error}"
    options.output.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(".cache/benchmark.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another benchmark is running in this checkout")
        write(options.output, artifact)
        width = len(options.engines) * len(MODES)
        for index, workload in enumerate(workloads):
            if workload.suite in errors:
                results = [
                    record(workload, engine, mode, "error", reason=errors[workload.suite])
                    for engine in options.engines
                    for mode in MODES
                ]
            else:
                results = run(
                    workload,
                    options.engines,
                    settings,
                    options.output.parent / "inputs",
                    options.rows,
                )
            artifact["results"][index * width : (index + 1) * width] = results
            write(options.output, artifact)
            print(
                f"{workload.identifier}: "
                + ", ".join(
                    f"{result['engine']}/{result['mode']}={result['status']}" for result in results
                ),
                flush=True,
            )
        artifact["complete"] = True
        write(options.output, artifact)
    print(f"Saved {len(artifact['results'])} results to {options.output}")
    return int(any(result["status"] == "error" for result in artifact["results"]))
