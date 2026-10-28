import hashlib
import importlib
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from bench.data import Input, prepare
from bench.engines.base import Adapter
from bench.fingerprint import fingerprint
from bench.timing import Settings, elapsed, paired, summarize
from bench.validation import validate
from udfx.backends.diagnostics import capture_lowering
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.program import Program
from workloads.models import Workload

ENGINES = ("polars", "duckdb", "datafusion", "spark")
MODES = ("memory", "query")


def record(workload: Workload, engine: str, mode: str, status: str, **values: object) -> dict:
    return {
        "workload": workload.identifier,
        "suite": workload.suite,
        "origin": workload.origin,
        "engine": engine,
        "mode": mode,
        "status": status,
        **values,
    }


def measure(
    adapter: Adapter, program: Program, frontend_seconds: float, settings: Settings
) -> list[dict]:
    workload = adapter.workload
    engine = adapter.backend.name
    data = adapter.data
    start = perf_counter()
    with capture_lowering() as details:
        expr = adapter.backend.build(program)
    compile_seconds = frontend_seconds + perf_counter() - start
    if not details:
        raise ValueError("The backend did not report its fallback guard")
    exact = details[-1].exact
    native_rows = (
        data.table.num_rows
        if exact is None
        else sum(value is True for value in adapter.execute(exact, "memory").column(1).to_pylist())
    )
    status = "python" if native_rows == 0 else "native" if exact is None else "guarded"
    expressions = {**adapter.python(), "udfx": expr}
    native = adapter.native()
    if native is not None:
        expressions["native"] = native
    common = {
        "validation": "exact" if workload.deterministic else "distribution",
        "compile_seconds": compile_seconds,
        "frontend_seconds": frontend_seconds,
        "native_rows": native_rows,
        "total_rows": data.table.num_rows,
        "native_row_fraction": native_rows / data.table.num_rows,
        "compiled_udfs": int(native_rows > 0),
        "total_udfs": 1,
        "data": data.metadata,
        "settings": adapter.settings(),
        "timing": asdict(settings),
        "primary_baseline": adapter.primary,
        "handwritten_native": "available" if native is not None else "not provided",
        "fingerprint": hashlib.sha256(fingerprint(adapter.backend, program).encode()).hexdigest(),
    }
    records = []
    for mode in MODES:
        try:
            actions = adapter.actions(expressions, mode)
            first = {}
            for name, action in actions.items():
                output, first[name] = elapsed(action)
                validate(output, data, mode, f"{workload.identifier}/{engine}/{name}")
            timing = paired(actions, settings)
            comparisons = {
                name: summarize(timing["samples"], name) for name in expressions if name != "udfx"
            }
            records.append(
                record(
                    workload,
                    engine,
                    mode,
                    status,
                    **common,
                    first_execution_seconds=first,
                    **timing,
                    comparisons=comparisons,
                )
            )
        except Exception as error:
            records.append(
                record(
                    workload,
                    engine,
                    mode,
                    "error",
                    **common,
                    reason=f"{type(error).__name__}: {error}",
                )
            )
    return records


def run_engine(
    workload: Workload,
    data: Input,
    program: Program,
    frontend_seconds: float,
    engine: str,
    *,
    settings: Settings,
) -> list[dict]:
    try:
        factory = importlib.import_module(f"bench.engines.{engine}").Engine
        adapter = factory(workload, data, settings.threads)
        try:
            return measure(adapter, program, frontend_seconds, settings)
        finally:
            adapter.close()
    except Exception as error:
        status = "unsupported" if isinstance(error, Unsupported) else "error"
        return [
            record(workload, engine, mode, status, reason=f"{type(error).__name__}: {error}")
            for mode in MODES
        ]


def run(
    workload: Workload,
    engines: list[str],
    settings: Settings,
    directory: Path,
    rows: int | None = None,
) -> list[dict]:
    if workload.error or workload.exclusion:
        status = "error" if workload.error else "excluded"
        return [
            record(workload, engine, mode, status, reason=workload.error or workload.exclusion)
            for engine in engines
            for mode in MODES
        ]
    try:
        if workload.function is None:
            raise ValueError("Missing Python function")
        start = perf_counter()
        program = trace(workload.function, workload.schema, result=workload.result)
        if not program.deterministic and workload.deterministic:
            raise ValueError("Nondeterministic workloads require distribution validation")
        frontend_seconds = perf_counter() - start
        data = prepare(workload, directory, rows)
    except Exception as error:
        status = "unsupported" if isinstance(error, Unsupported) else "error"
        return [
            record(workload, engine, mode, status, reason=f"{type(error).__name__}: {error}")
            for engine in engines
            for mode in MODES
        ]
    return [
        result
        for engine in engines
        for result in run_engine(
            workload, data, program, frontend_seconds, engine, settings=settings
        )
    ]
