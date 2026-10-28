import random

import pytest

from bench.distributions import verify_distribution


def test_distribution_validation_keeps_independent_samples() -> None:
    generator = random.Random(42)
    expected = [[index, generator.gauss(0, 1)] for index in range(2000)]
    actual = [[index, generator.gauss(0, 1)] for index in range(2000)]
    verify_distribution(actual, expected, "independent")
    with pytest.raises(AssertionError, match="distributions differ"):
        verify_distribution([[index, value + 2] for index, value in actual], expected, "shift")
    with pytest.raises(AssertionError, match="distributions differ"):
        verify_distribution([[index, 0.0] for index in range(2000)], expected, "constant")


@pytest.mark.parametrize(
    "actual",
    [[[0, 1.0]], [[0, 1.0], [0, None]], [[0, None], [1, 1.0]]],
)
def test_distribution_validation_preserves_rows_and_nulls(actual: list[list]) -> None:
    with pytest.raises(AssertionError, match="row identities or null"):
        verify_distribution(actual, [[0, 1.0], [1, None]], "shape")


def test_distribution_validation_preserves_grouped_counts() -> None:
    expected = [["a", 1.0, 3], ["b", 2.0, 4]]
    verify_distribution([["b", 2.1, 4], ["a", 1.1, 3]], expected, "grouped")
    with pytest.raises(AssertionError, match="counts differ"):
        verify_distribution([["a", 1.0, 4], ["b", 2.0, 4]], expected, "count")
    with pytest.raises(AssertionError, match="groups differ"):
        verify_distribution([["a", 1.0, 7]], expected, "group")
