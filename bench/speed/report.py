import os
from collections.abc import Mapping, Sequence
from hashlib import sha256
from statistics import geometric_mean

from bench.speed.timing import Measurement

K3_TARGETS = {"spark": 1.0}


def report(
    measurements: Sequence[Measurement], skipped: Mapping[str, str], row_count: int, repeats: int
) -> str:
    load = " ".join(f"{value:.2f}" for value in os.getloadavg())
    lines = [
        f"rows {row_count}, best of {repeats}, load average {load}\n",
        "| case | engine | python form | python s | udfx s | speedup | fallback in plan"
        " | pending share | K2 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in sorted(measurements, key=lambda item: item.speedup):
        verdict = "PASS" if item.speedup >= 1.0 else "FAIL"
        lines.append(
            f"| {item.case} | {item.engine} | {item.form} | {item.python:.6f} | {item.native:.6f}"
            f" | {item.speedup:.2f}x | {str(item.fallback).lower()}"
            f" | {item.pending:.2%} | {verdict} |"
        )
    lines.extend(
        [
            "\n| engine | cases | groups | K2 fails | geomean | deduplicated geomean | K3"
            " | K5 no fallback |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    duplicates = [
        "\n| engine | fingerprint | cases | speedups |",
        "|---|---|---|---|",
    ]
    for engine in dict.fromkeys(item.engine for item in measurements):
        rows = [item for item in measurements if item.engine == engine]
        groups: dict[str, list[Measurement]] = {}
        for item in rows:
            groups.setdefault(item.fingerprint, []).append(item)
        speedups = [item.speedup for item in rows]
        native_only = sum(not item.fallback for item in rows)
        fails = sum(speedup < 1.0 for speedup in speedups)
        mean = geometric_mean(speedups)
        deduplicated = geometric_mean(
            geometric_mean(item.speedup for item in group) for group in groups.values()
        )
        verdict = "PASS" if deduplicated >= K3_TARGETS.get(engine, 3.0) else "FAIL"
        lines.append(
            f"| {engine} | {len(rows)} | {len(groups)} | {fails} | {mean:.2f}x"
            f" | {deduplicated:.2f}x | {verdict} | {native_only}/{len(rows)} |"
        )
        for fingerprint, group in groups.items():
            if len(group) > 1:
                digest = sha256(fingerprint.encode()).hexdigest()[:12]
                names = ", ".join(item.case for item in group)
                values = ", ".join(f"{item.speedup:.2f}x" for item in group)
                duplicates.append(f"| {engine} | {digest} | {names} | {values} |")
    lines.extend(duplicates)
    for name, reason in skipped.items():
        lines.append(f"\nskipped {name}: {reason}")
    return "\n".join(lines)
