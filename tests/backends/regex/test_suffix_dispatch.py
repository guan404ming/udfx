from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.engines.polars import PolarsBackend
from udfx.backends.lowering.programs import factored
from udfx.backends.rewrites.factoring import factor_branches
from udfx.backends.rewrites.suffixes import dispatch_suffixes
from udfx.frontend import trace
from udfx.graphs import parent_names
from udfx.ir import Col, If, lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import BOOL, STR, Kind, Type

ROWS = [
    ("",),
    ("rational",),
    ("balance",),
    ("silence",),
    ("baker",),
    ("national",),
    ("tional",),
    ("sizer",),
    ("boss",),
    ("bus",),
    ("é",),
]


def replace_suffix(word: str) -> str:
    if word.endswith("ational"):
        return word[: len(word) - 7] + "ate"
    if word.endswith("tional"):
        return word[: len(word) - 6] + "tion"
    if word.endswith("izer"):
        return word[: len(word) - 4] + "ize"
    return word


def merged_suffixes(word: str) -> str:
    if word.endswith("ance") or word.endswith("ence"):
        return word[: len(word) - 4] + "e"
    if word.endswith("er"):
        return word[: len(word) - 2] + "e"
    return word


def shorter_suffix_first(word: str) -> str:
    if word.endswith("s"):
        return word[: len(word) - 1]
    if word.endswith("ss"):
        return word[: len(word) - 2]
    return word


def operations(function: Any) -> frozenset[str]:
    program = factored(PolarsBackend(), trace(function, {"word": STR}))
    return frozenset().union(*parent_names((program.value,)).values())


def test_suffix_chain_uses_one_search() -> None:
    assert "re.search" in operations(replace_suffix)
    assert "re.search" in operations(merged_suffixes)
    assert "re.search" not in operations(shorter_suffix_first)


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [replace_suffix, merged_suffixes, shorter_suffix_first])
def test_suffix_dispatch_matches_python(backend: Backend[Any], function: Any) -> None:
    assert check_function(function, {"word": STR}, backend, ROWS) == []


def test_nullable_receiver_uses_one_search() -> None:
    word = Col("word", Type(Kind.STR, nullable=True))
    node = If(
        apply(".endswith", word, lit("ational")),
        apply("__add__", word, lit("ate")),
        If(
            apply(".endswith", word, lit("izer")),
            apply("__add__", word, lit("ize")),
            word,
            word.type,
        ),
        word.type,
    )
    (dispatched,) = dispatch_suffixes((node,), lambda call, index: False)
    assert "re.search" in frozenset().union(*parent_names((dispatched,)).values())


def test_branch_factoring_moves_condition_into_argument() -> None:
    condition = Col("flag", BOOL)
    left, right = Col("left", STR), Col("right", STR)
    node = If(
        condition,
        apply(".upper", apply(".strip", left)),
        apply(".upper", apply(".strip", right)),
        STR,
    )
    (factored_node,) = factor_branches((node,), lambda call, index: False)
    assert factored_node == apply(".upper", apply(".strip", If(condition, left, right, STR)))
