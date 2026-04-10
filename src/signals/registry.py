"""Signal registry with auto-discovery.

Scans src/signals/ at import time for modules that export a class inheriting
from BaseSignalSource, instantiates them, and exposes them via `all_sources()`.
New sources drop into the folder and become active on the next Dagster run
without touching any code outside the module.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Iterable

import structlog

from src.signals.base import BaseSignalSource

log = structlog.get_logger(__name__)

_SOURCES: dict[str, BaseSignalSource] = {}


def _discover() -> None:
    """Walk src/signals/ and import every submodule, picking up subclasses."""
    import src.signals as pkg

    for modinfo in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}."):
        if modinfo.name in (f"{pkg.__name__}.base", f"{pkg.__name__}.registry"):
            continue
        try:
            mod = importlib.import_module(modinfo.name)
        except Exception as exc:  # noqa: BLE001
            log.warning("registry.import_failed", module=modinfo.name, error=str(exc))
            continue
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseSignalSource)
                and attr is not BaseSignalSource
            ):
                try:
                    instance = attr()
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "registry.instantiate_failed",
                        cls=attr.__name__,
                        error=str(exc),
                    )
                    continue
                _SOURCES[instance.source_name] = instance
                log.info("registry.registered", source=instance.source_name)


def all_sources() -> dict[str, BaseSignalSource]:
    if not _SOURCES:
        _discover()
    return dict(_SOURCES)


def get_source(name: str) -> BaseSignalSource:
    sources = all_sources()
    if name not in sources:
        raise KeyError(f"Signal source '{name}' not registered. Available: {list(sources)}")
    return sources[name]


def source_names() -> Iterable[str]:
    return all_sources().keys()
