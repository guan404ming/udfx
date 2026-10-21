from __future__ import annotations

SET_ORDER_MAX_SIZE = 1


def set_size(values: list[str]) -> int:
    return len(set(values))


def set_list(values: list[str]) -> list[str]:
    return list(set(values))


def sorted_set(values: list[str]) -> list[str]:
    return sorted(set(values))
