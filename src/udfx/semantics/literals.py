from __future__ import annotations

ASCII_SPACE = "\t\n\x0b\x0c\r "
DIGITS = "[0-9](_?[0-9])*"
INT_LITERAL = f"^[{ASCII_SPACE}]*[+-]?{DIGITS}[{ASCII_SPACE}]*$"
FLOAT_LITERAL = (
    f"(?i)^[{ASCII_SPACE}]*[+-]?"
    f"(({DIGITS}(\\.({DIGITS})?)?|\\.{DIGITS})(e[+-]?{DIGITS})?|inf|infinity|nan)"
    f"[{ASCII_SPACE}]*$"
)
