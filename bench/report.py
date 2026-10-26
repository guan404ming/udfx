import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from bench.artifacts import read


def geometric_mean(values: list[float]) -> float | None:
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def speedups(results: list[dict]) -> tuple[list[float], float | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    ratios = []
    for result in results:
        if result["status"] in {"native", "guarded"}:
            ratio = result["comparisons"][result["primary_baseline"]]["median_speedup"]
            ratios.append(ratio)
            groups[result.get("fingerprint", result["workload"])].append(ratio)
    means = [mean for values in groups.values() if (mean := geometric_mean(values)) is not None]
    return ratios, geometric_mean(means)


def markdown(artifact: dict) -> str:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for result in artifact["results"]:
        groups[result["suite"], result["engine"], result["mode"]].append(result)
    lines = [
        "# Benchmark results",
        "",
        f"Commit: `{artifact['commit']}`. Python: `{artifact['python'].split()[0]}`.",
        f"Run complete: {artifact.get('complete', False)}.",
        "",
        "Speedups use the declared Python baseline and the median of paired rounds.",
        "Native and guarded counts require correct output. Errors remain in the denominator.",
        "",
        "| Suite | Engine | Mode | Native | Guarded | Unsupported | Errors | Excluded |"
        " Pending | Geomean | Dedup | Slower |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (suite, engine, mode), results in sorted(groups.items()):
        counts = Counter(result["status"] for result in results)
        ratios, dedup = speedups(results)
        mean = f"{geometric_mean(ratios):.2f}x" if ratios else "N/A"
        grouped = f"{dedup:.2f}x" if dedup is not None else "N/A"
        lines.append(
            f"| {suite} | {engine} | {mode} | {counts['native']} | {counts['guarded']} | "
            f"{counts['unsupported']} | {counts['error']} | {counts['excluded']} | "
            f"{counts['pending']} | {mean} | {grouped} | "
            f"{sum(ratio < 1 for ratio in ratios)}/{len(ratios)} |"
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Workload | Engine | Mode | Status | Baseline | Speedup | Native rows |",
            "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for result in artifact["results"]:
        comparisons = result.get("comparisons", {})
        share = f"{result['native_row_fraction']:.1%}" if "native_row_fraction" in result else "N/A"
        for baseline, comparison in comparisons.items() or [("N/A", {})]:
            speedup = f"{comparison['median_speedup']:.2f}x" if comparison else "N/A"
            lines.append(
                f"| {result['workload']} | {result['engine']} | {result['mode']} | "
                f"{result['status']} | {baseline} | {speedup} | {share} |"
            )
    failures = [result for result in artifact["results"] if "reason" in result]
    if failures:
        lines.extend(["", "## Reasons", ""])
        reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
        for result in failures:
            reasons[result["workload"], result["reason"]].add(result["engine"])
        for (name, reason), engines in sorted(reasons.items()):
            lines.append(
                f"- `{name}` ({', '.join(sorted(engines))}): {reason.replace(chr(10), ' ')}"
            )
    return "\n".join(lines) + "\n"


def export_csv(artifact: dict, path: Path) -> None:
    names = [
        "workload",
        "engine",
        "mode",
        "status",
        "baseline",
        "median_speedup",
        "median_absolute_deviation",
        "native_row_fraction",
        "reason",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for result in artifact["results"]:
            for baseline, comparison in result.get("comparisons", {}).items() or [("", {})]:
                row = {name: result.get(name, comparison.get(name, "")) for name in names}
                writer.writerow({**row, "baseline": baseline})


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render a saved benchmark artifact without rerunning it"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--csv", type=Path)
    options = parser.parse_args(arguments)
    artifact = read(options.artifact)
    if options.markdown:
        options.markdown.write_text(markdown(artifact))
    if options.csv:
        export_csv(artifact, options.csv)
    if not options.markdown and not options.csv:
        print(markdown(artifact), end="")


if __name__ == "__main__":
    main()
