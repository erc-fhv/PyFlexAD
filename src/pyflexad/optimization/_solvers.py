"""Internal solver access helpers.

This module centralizes the logic for importing and checking availability of optional solvers.
Extend this module to support additional backends (e.g., HiGHS) in the future.
"""
from __future__ import annotations

from typing import Any

__all__ = ["import_gurobi", "is_gurobi_available"]


def import_gurobi() -> Any:
    """Import and return the Gurobi Python module.

    Raises
    ------
    ImportError
        If Gurobi is not installed or cannot be imported. The error message
        explains how to install the optional dependency and reminds about licensing.
    """
    try:
        import gurobipy as gp  # type: ignore
    except Exception as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "Gurobi is required for this optimization routine. Install with 'pip install pyflexad[gurobi]' "
            "and ensure a valid Gurobi license is available. See https://www.gurobi.com for details."
        ) from e
    return gp


def is_gurobi_available() -> bool:
    """Return True if Gurobi can be imported, else False."""
    try:
        import gurobipy  # noqa: F401
        return True
    except Exception:  # pragma: no cover - depends on environment
        return False
