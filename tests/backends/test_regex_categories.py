import pytest

from udfx.backends.spark import SparkBackend
from udfx.backends.spark.regex_categories import category, native_category
from udfx.ir import Col
from udfx.ops import apply
from udfx.semantics.regex_categories import predicate_for
from udfx.semantics.regex_translation import Translator, translate
from udfx.semantics.text import code_points
from udfx.types import STR

pytestmark = pytest.mark.xdist_group("spark")


@pytest.mark.parametrize("name", ["CATEGORY_DIGIT", "CATEGORY_SPACE", "CATEGORY_WORD"])
@pytest.mark.parametrize("negated", [False, True])
@pytest.mark.parametrize("ascii_only", [False, True])
def test_java_categories_preserve_python_unicode(
    name: str, negated: bool, ascii_only: bool
) -> None:
    translator = Translator(0, "java", ascii_only, category)
    pattern = translator.category(name.replace("CATEGORY_", "CATEGORY_NOT_") if negated else name)
    predicate = predicate_for(name)
    expected = "".join(
        point
        for point in code_points()
        if predicate(point) != negated and (point.isascii() or not ascii_only)
    )
    assert native_category(pattern) == expected


def test_java_word_boundary_uses_compact_categories() -> None:
    pattern = translate(r"\b\w+\b", 0, "java", categories=category).source
    assert r"\p{L}" in pattern
    assert pattern.count(category("CATEGORY_WORD")) == 6


def test_java_integer_parser_uses_corrected_category() -> None:
    lowered = SparkBackend().lower(apply("int_parses", Col("value", STR)))
    assert "p{Nd}" in str(lowered.expr)
