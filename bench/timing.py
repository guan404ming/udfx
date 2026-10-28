import gc
import math
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class Settings:
    warmups: int = 2
    rounds: int = 7
    threads: int = 1

    def __post_init__(self) -> None:
        if self.warmups < 0 or self.rounds < 1 or self.threads < 1:
            raise ValueError("Warmups must be non-negative; rounds and threads must be positive")


def elapsed[Result](action: Callable[[], Result]) -> tuple[Result, float]:
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = perf_counter()
        result = action()
        duration = perf_counter() - start
    finally:
        if enabled:
            gc.enable()
    return result, duration


def balanced_orders(names: list[str]) -> list[list[str]]:
    count = len(names)
    positions = [0] + [
        (index + 1) // 2 if index % 2 else count - index // 2 for index in range(1, count)
    ]
    orders = [[names[(index + offset) % count] for index in positions] for offset in range(count)]
    return orders if count % 2 == 0 else orders + [order[::-1] for order in orders]


def paired(actions: Mapping[str, Callable[[], object]], settings: Settings) -> dict:
    schedule = balanced_orders(list(actions))
    samples = []
    orders = []
    for iteration in range(settings.warmups + settings.rounds):
        order = schedule[(iteration - settings.warmups) % len(schedule)]
        sample = {name: elapsed(actions[name])[1] for name in order}
        if iteration >= settings.warmups:
            samples.append(sample)
            orders.append(order)
    return {"samples": samples, "orders": orders}


def summarize(samples: list[dict[str, float]], baseline: str) -> dict[str, float]:
    if not samples:
        raise ValueError("At least one paired sample is required")
    if not all(
        math.isfinite(sample[name]) and sample[name] > 0
        for sample in samples
        for name in (baseline, "udfx")
    ):
        raise ValueError("Timings must be finite and positive")
    ratios = [sample[baseline] / sample["udfx"] for sample in samples]
    median = statistics.median(ratios)
    return {
        "median_speedup": median,
        "median_absolute_deviation": statistics.median(abs(value - median) for value in ratios),
        "minimum_speedup": min(ratios),
        "maximum_speedup": max(ratios),
        "baseline_median_seconds": statistics.median(sample[baseline] for sample in samples),
        "udfx_median_seconds": statistics.median(sample["udfx"] for sample in samples),
    }
