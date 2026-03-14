from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "ProviderBundle",
    "RuntimeConfig",
    "RuntimeContext",
    "build_provider_bundle",
    "create_runtime",
    "load_anchor_set",
    "run_pipeline",
]


def __getattr__(name: str) -> Any:
    if name in {"ProviderBundle", "build_provider_bundle"}:
        module = import_module(".providers", __name__)
        return getattr(module, name)
    if name in {"create_runtime", "load_anchor_set", "run_pipeline"}:
        module = import_module(".qa_pipeline", __name__)
        return getattr(module, name)
    if name in {"RuntimeConfig", "RuntimeContext"}:
        module = import_module(".qa_runtime", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
