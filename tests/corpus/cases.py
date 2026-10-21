from . import Case
from .cases_numbers import CASES as NUMBERS
from .cases_rows import CASES as ROWS
from .cases_strings import CASES as STRINGS
from .cases_structure import CASES as STRUCTURE

CASES = [*STRINGS, *NUMBERS, *STRUCTURE, *ROWS]

__all__ = ["CASES", "Case"]
