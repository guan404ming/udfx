from . import Case
from .cases_numbers import CASES as NUMBERS
from .cases_strings import CASES as STRINGS
from .cases_structure import CASES as STRUCTURE

CASES = [*STRINGS, *NUMBERS, *STRUCTURE]

__all__ = ["CASES", "Case"]
