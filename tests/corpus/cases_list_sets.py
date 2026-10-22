from udfx.types import STR

from . import Case


def jaccard(left: str, right: str) -> float:
    first = set(left.split())
    second = set(right.split())
    return len(first & second) / len(first | second)


def set_membership(left: str, right: str) -> bool:
    return "a" in (set(left.split()) ^ set(right.split()))


def set_chained(left: str, right: str) -> list[str]:
    return sorted((set(left) | set(right)) - (set(left) & set(right)))


def set_iterable_methods(left: str, right: str) -> int:
    return len(set(left).intersection().union(right, left.split()).difference(right.split()))


CASES = [
    Case(function.__name__, function, {"left": STR.or_none(), "right": STR.or_none()}, "native")
    for function in (
        jaccard,
        set_membership,
        set_chained,
        set_iterable_methods,
    )
]
