import math
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence


def samples(rows: Sequence[Sequence[object]]) -> dict[object, list[float]]:
    grouped: dict[object, list[float]] = defaultdict(list)
    for row in rows:
        key, value, *counts = row
        if value is None:
            continue
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise AssertionError("Distribution validation requires finite numeric output")
        count = counts[0] if counts else 1
        if not isinstance(count, int) or count < 1:
            raise AssertionError("Distribution counts must be positive integers")
        grouped[key if counts else None].extend([float(value)] * count)
    return grouped


def verify_distribution(
    actual: Sequence[Sequence[object]], expected: Sequence[Sequence[object]], label: str
) -> None:
    def shape(rows: Sequence[Sequence[object]]) -> list[tuple[object, bool]]:
        return sorted(((row[0], row[1] is None) for row in rows), key=repr)

    if (not expected or len(expected[0]) == 2) and shape(actual) != shape(expected):
        raise AssertionError(f"{label}: row identities or null outputs differ")
    left, right = samples(actual), samples(expected)
    if left.keys() != right.keys():
        raise AssertionError(f"{label}: output groups differ")
    for key in right:
        observed, reference = sorted(left[key]), sorted(right[key])
        if len(observed) != len(reference):
            raise AssertionError(f"{label}: output counts differ in {key!r}")
        count = len(reference)
        bound = math.sqrt(2 * math.log(4e9 * max(len(right), 1)) / count)
        distance = max(
            abs(bisect_right(observed, value) - bisect_right(reference, value)) / count
            for value in set(observed) | set(reference)
        )
        if distance > bound:
            raise AssertionError(f"{label}: output distributions differ in {key!r}")
