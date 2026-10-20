from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from udfx.backends.base import Backend


def known() -> list[str]:
    return sorted(module.name for module in pkgutil.iter_modules(__path__) if module.ispkg)


def named(name: str) -> Backend[Any]:
    names = known()
    if name not in names:
        raise ValueError(f"unknown backend {name!r}; known backends: {', '.join(names)}")
    try:
        module = importlib.import_module(f"udfx.backends.{name}")
    except ImportError as error:
        raise ImportError(
            f"backend {name!r} needs its engine: pip install 'udfx[{name}]'"
        ) from error
    backend: type[Backend[Any]] = module.BACKEND
    return backend()


def installed() -> tuple[Backend[Any], ...]:
    found: list[Backend[Any]] = []
    for name in known():
        try:
            found.append(named(name))
        except Exception:
            continue
    return tuple(found)
