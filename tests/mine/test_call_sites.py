import ast

import pytest

from bench.mine.call_sites import find_candidates


@pytest.mark.parametrize(
    ("source", "sites"),
    [
        (f"frame.{name}(lambda value: value, lambda ignored: ignored)", [f".{name}"])
        for name in ("apply", "map", "applymap", "transform", "agg", "map_elements", "map_rows")
    ]
    + [
        ("udf(lambda value: value)", ["udf"]),
        ("functions.udf(function)", ["udf"]),
        ("pandas_udf(f=lambda value: value)", ["pandas_udf"]),
        ("np.vectorize(function)", ["np.vectorize"]),
        ("@udf\ndef function(value): return value", ["@udf"]),
        ("@pandas_udf(schema)\ndef function(value): return value", ["@pandas_udf"]),
        ("frame.agg(['sum'])\nframe.apply('sum')\nother.vectorize(function)", []),
    ],
)
def test_candidate_sites(source: str, sites: list[str]) -> None:
    assert [site for site, _, _ in find_candidates(ast.parse(source))] == sites


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("frame.apply(function)", "element"),
        ("frame.apply(func=function, axis=0)", "element"),
        ("frame.map(function)", "element"),
        ("frame.applymap(function)", "element"),
        ("frame.map_elements(function)", "element"),
        ("np.vectorize(function, signature=signature)", "element"),
        ("udf(function, returnType=schema)", "element"),
        ("frame.apply(function, axis=1)", "row"),
        ("frame.apply(function, axis='columns')", "row"),
        ("frame.map_rows(function)", "row"),
        ("frame.transform(function)", "group"),
        ("frame.agg(function)", "group"),
        ("frame.aggregate(function)", "group"),
        ("pandas_udf(function)", "group"),
        ("@pandas_udf(schema)\ndef function(value): return value", "group"),
        *[
            (f"frame.{method}('key')['value'].apply(function, axis=1)", "group")
            for method in ("groupby", "rolling", "expanding", "resample", "ewm")
        ],
    ],
)
def test_candidate_kinds(source: str, kind: str) -> None:
    found = find_candidates(ast.parse(source))
    assert len(found) == 1
    assert found[0][1] == kind
