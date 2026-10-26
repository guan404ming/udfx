import gc
from collections.abc import Callable

import pytest

from bench.timing import Settings, elapsed, paired, summarize


@pytest.mark.parametrize("names", [("python", "udfx"), ("python", "udfx", "native")])
def test_pairing_retains_every_round_and_changes_order(
    monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]
) -> None:
    calls: list[str] = []
    ticks = iter(range(100))
    monkeypatch.setattr("bench.timing.perf_counter", lambda: next(ticks))
    actions = {name: lambda name=name: calls.append(name) for name in names}
    result = paired(actions, Settings(warmups=2, rounds=7))
    assert len(calls) == 9 * len(names)
    assert len(result["samples"]) == 7
    assert all(sample == dict.fromkeys(actions, 1) for sample in result["samples"])
    assert len({tuple(order) for order in result["orders"]}) > 1


def test_speedup_is_median_of_pairs_not_ratio_of_medians() -> None:
    samples = [
        {"python": 10.0, "udfx": 1.0},
        {"python": 20.0, "udfx": 100.0},
        {"python": 30.0, "udfx": 20.0},
    ]
    result = summarize(samples, "python")
    assert result["median_speedup"] == 1.5
    assert result["minimum_speedup"] == 0.2
    assert result["median_absolute_deviation"] == 1.3


@pytest.mark.parametrize("enabled", [True, False])
def test_elapsed_restores_collection_after_failure(enabled: bool) -> None:
    original = gc.isenabled()
    gc.enable() if enabled else gc.disable()

    def action() -> None:
        assert not gc.isenabled()
        raise ValueError("failure")

    try:
        with pytest.raises(ValueError, match="failure"):
            elapsed(action)
        assert gc.isenabled() == enabled
    finally:
        gc.enable() if original else gc.disable()


@pytest.mark.parametrize("arguments", [{"warmups": -1}, {"rounds": 0}, {"threads": 0}])
def test_invalid_timing_settings(arguments: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        Settings(**arguments)


@pytest.mark.parametrize("samples", [[], [{"python": float("inf"), "udfx": 1.0}]])
def test_summary_rejects_invalid_samples(samples: list[dict[str, float]]) -> None:
    with pytest.raises(ValueError):
        summarize(samples, "python")


def test_baselines_run_again_on_each_measurement() -> None:
    calls: list[int] = []
    actions: dict[str, Callable[[], object]] = {
        "python": lambda: calls.append(1),
        "udfx": lambda: 0,
    }
    for _ in range(2):
        paired(actions, Settings(warmups=0, rounds=1))
    assert calls == [1, 1]
