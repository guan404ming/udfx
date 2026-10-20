from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from udfx.backends.base import Backend


def named(name: str) -> Backend[Any]:
    module = importlib.import_module(f"udfx.backends.{name}")
    backend: type[Backend[Any]] = module.BACKEND
    return backend()


def installed() -> tuple[Backend[Any], ...]:
    found: list[Backend[Any]] = []
    for module in pkgutil.iter_modules(__path__):
        if not module.ispkg:
            continue
        try:
            found.append(named(module.name))
        except ImportError:
            continue
    return tuple(found)
