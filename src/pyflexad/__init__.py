"""
pyflexad: Python Flexibility Aggregation and Disaggregation.

This package exposes the main subpackages at the top level and provides the package version.
The subpackages are lazily imported to keep import-time overhead minimal.
"""
from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _version
from typing import TYPE_CHECKING

try:
    __version__ = _version("pyflexad")
except PackageNotFoundError:  # pragma: no cover - when running from source without installed dist
    __version__ = "0.0.0"

# Public API surface at the package root
__all__ = [
    "math",
    "utils",
    "models",
    "system",
    "virtual",
    "physical",
    "parameters",
    "optimization",
]


def __getattr__(name: str):
    """Lazily import top-level subpackages upon attribute access.

    This keeps "import pyflexad" lightweight and avoids importing heavy optional
    dependencies until they are actually needed by the user.
    """
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Help static type checkers understand available attributes without triggering imports at runtime
if TYPE_CHECKING:  # pragma: no cover - type checking only
    from . import math as math
    from . import utils as utils
    from . import models as models
    from . import system as system
    from . import virtual as virtual
    from . import physical as physical
    from . import parameters as parameters
    from . import optimization as optimization
