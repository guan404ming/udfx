from . import Case
from .cases_formatting import CASES as FORMATTING
from .cases_lists import CASES as LISTS
from .cases_numbers import CASES as NUMBERS
from .cases_regex import CASES as REGEX
from .cases_rows import CASES as ROWS
from .cases_strings import CASES as STRINGS
from .cases_structure import CASES as STRUCTURE

CASES = [*STRINGS, *NUMBERS, *STRUCTURE, *ROWS, *FORMATTING, *REGEX, *LISTS]

__all__ = ["CASES", "Case"]
