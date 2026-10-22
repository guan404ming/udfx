import math
import os
from collections.abc import Mapping, Sequence

from bench.speed.timing import Measurement


def report(
    measurements: Sequence[Measurement], skipped: Mapping[str, str], row_count: int, repeats: int
) -> str:
    load = " ".join(f"{value:.2f}" for value in os.getloadavg())
    lines = [
        f"rows {row_count}, best of {repeats}, load average {load}\n",
        "| case | engine | python form | python s | udfx s | speedup | fallback in plan | K2 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in sorted(measurements, key=lambda item: item.speedup):
        verdict = "PASS" if item.speedup >= 1.0 else "FAIL"
        lines.append(
            f"| {item.case} | {item.engine} | {item.form} | {item.python:.6f} | {item.native:.6f}"
            f" | {item.speedup:.2f}x | {str(item.fallback).lower()} | {verdict} |"
        )
    lines.extend(
        [
            "\n| engine | cases | K2 fails | geomean | K3 | K5 no fallback |",
            "|---|---|---|---|---|---|",
        ]
    )
    for engine in dict.fromkeys(item.engine for item in measurements):
        rows = [item for item in measurements if item.engine == engine]
        speedups = [item.speedup for item in rows]
        native_only = sum(not item.fallback for item in rows)
        fails = sum(speedup < 1.0 for speedup in speedups)
        mean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
        verdict = "PASS" if mean >= 3.0 else "FAIL"
        lines.append(
            f"| {engine} | {len(rows)} | {fails} | {mean:.2f}x | {verdict}"
            f" | {native_only}/{len(rows)} |"
        )
    for name, reason in skipped.items():
        lines.append(f"\nskipped {name}: {reason}")
    return "\n".join(lines)
