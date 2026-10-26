from . import Case
from .cases_datetimes import CASES as DATETIMES
from .cases_formatting import CASES as FORMATTING
from .cases_lists import CASES as LISTS
from .cases_numbers import CASES as NUMBERS
from .cases_paths import CASES as PATHS
from .cases_regex import CASES as REGEX
from .cases_rows import CASES as ROWS
from .cases_small_gaps import CASES as SMALL_GAPS
from .cases_strings import CASES as STRINGS
from .cases_structure import CASES as STRUCTURE
from .cases_translation import CASES as TRANSLATION

CASES = [
    *STRINGS,
    *NUMBERS,
    *STRUCTURE,
    *ROWS,
    *FORMATTING,
    *REGEX,
    *LISTS,
    *SMALL_GAPS,
    *DATETIMES,
    *PATHS,
    *TRANSLATION,
]

__all__ = ["CASES", "Case"]
