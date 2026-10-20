from __future__ import annotations


class Unsupported(Exception):
    def __init__(self, reason: str, where: str = "") -> None:
        super().__init__(f"{reason} at {where}" if where else reason)
        self.reason = reason
        self.where = where
