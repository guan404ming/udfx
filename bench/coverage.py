import argparse
import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from bench.artifacts import write
from bench.runner import ENGINES
from udfx.ops import all_ops


def code_summary(report: dict) -> dict:
    groups: dict[str, Counter] = {"compiler": Counter(), "framework": Counter()}
    for filename, record in report["files"].items():
        if filename.startswith("src/udfx/"):
            group = "compiler"
        elif filename.startswith(("bench/", "workloads/")) and not filename.startswith(
            ("bench/mine/", "workloads/corpus/")
        ):
            group = "framework"
        else:
            continue
        summary = record["summary"]
        groups[group].update(
            {
                name: summary[name]
                for name in ("covered_lines", "num_statements", "covered_branches", "num_branches")
            }
        )
    return {
        group: {
            **counts,
            "line_percent": 100 * counts["covered_lines"] / counts["num_statements"]
            if counts["num_statements"]
            else None,
            "branch_percent": 100 * counts["covered_branches"] / counts["num_branches"]
            if counts["num_branches"]
            else None,
        }
        for group, counts in groups.items()
    }


def semantic_summary(paths: list[Path]) -> dict:
    observed: dict[tuple[str, str], set[str]] = {}
    dimensions: dict[tuple[str, str], set[str]] = {}
    for path in paths:
        for case in ElementTree.parse(path).iter("testcase"):
            properties = {
                item.attrib["name"]: item.attrib["value"]
                for item in case.findall("properties/property")
            }
            if "signature" not in properties:
                continue
            key = (properties["backend"], properties["signature"])
            state = (
                "failed"
                if case.find("failure") is not None or case.find("error") is not None
                else "skipped"
                if case.find("skipped") is not None
                else "passed"
            )
            observed.setdefault(key, set()).add(state)
            dimensions.setdefault(key, set()).update(properties.get("dimensions", "").split(","))
    matrix = []
    for engine in ENGINES:
        for op in all_ops():
            key = (engine, str(op))
            states = observed.get(key, set())
            status = (
                "failed"
                if "failed" in states
                else "passed"
                if "passed" in states
                else "skipped"
                if states
                else "not_run"
            )
            matrix.append(
                {
                    "backend": engine,
                    "signature": str(op),
                    "status": status,
                    "dimensions": sorted(dimensions.get(key, set()) - {""}),
                }
            )
    return {"counts": dict(Counter(row["status"] for row in matrix)), "matrix": matrix}


def check_baseline(summary: dict, baseline: dict) -> list[str]:
    errors = []
    for group, thresholds in baseline.items():
        for metric, minimum in thresholds.items():
            actual = summary[group][metric]
            if actual is None or actual < minimum:
                errors.append(f"{group}.{metric}: {actual} < {minimum}")
    return errors


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine code coverage with executed semantic signatures"
    )
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--junit", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    code = code_summary(json.loads(options.code.read_text()))
    semantic = semantic_summary(options.junit)
    write(options.output, {"schema_version": 1, "code": code, "semantic": semantic})
    errors = (
        check_baseline(code, json.loads(options.baseline.read_text())) if options.baseline else []
    )
    print(json.dumps({"code": code, "semantic": semantic["counts"], "errors": errors}, indent=2))
    return int(
        bool(errors)
        or any(semantic["counts"].get(state, 0) for state in ("failed", "skipped", "not_run"))
    )


if __name__ == "__main__":
    raise SystemExit(main())
