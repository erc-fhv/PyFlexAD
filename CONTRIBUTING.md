# Contributing to PyFlexAD

Thank you for considering contributing to PyFlexAD! This document describes how to set up a development environment, run tests, lint the codebase, and build distributions. It also explains how to work with optional solvers.

## Getting Started

- Python: 3.11–3.12 (see `requires-python` in pyproject.toml)
- Build backend: Hatchling
- Source layout: `src/pyflexad`

Clone the repository:

```bash
git clone https://github.com/erc-fhv/PyFlexAD.git
cd PyFlexAD
```

### Install (editable) and dev tools

Using pip:

```bash
python -m venv .venv
# Activate your venv, e.g. on Windows PowerShell:
#   .venv\Scripts\Activate.ps1
# On bash:
#   source .venv/bin/activate

pip install -e .
# Install dev tools (pytest, ruff, etc.)
pip install '.[dev]'
```

Using uv (optional):

```bash
uv venv
# Activate the venv as above
uv pip install -e .
uv pip install -r pyproject.toml#dev
```

### Optional solver dependencies

Install optional solvers as needed:

- Gurobi (commercial, license required):
  ```bash
  pip install 'pyflexad[gurobi]'
  ```
  You must have a valid Gurobi license. See https://www.gurobi.com/ for details.

- HiGHS (open source):
  ```bash
  pip install 'pyflexad[highs]'
  ```

- All optional solvers:
  ```bash
  pip install 'pyflexad[all-solvers]'
  ```

Note: The codebase defers solver imports to runtime. If you use a controller that requires Gurobi without having it installed/licensed, you will receive a clear ImportError.

## Running Tests

The test suite is configured via `pyproject.toml`:

```bash
pytest
```

Coverage report:

```bash
pytest --cov --cov-report html --cov-report term-missing --cov-fail-under 75
```

Parallel tests (if desired, requires `pytest-xdist`):

```bash
pytest -n auto
```

If you introduce tests that depend on Gurobi, please guard them to skip when the solver is unavailable, for example:

```python
import pytest
from pyflexad.optimization._solvers import is_gurobi_available

requires_gurobi = pytest.mark.skipif(
    not is_gurobi_available(),
    reason="Gurobi not available; install with pyflexad[gurobi] and ensure a valid license.",
)

@requires_gurobi
def test_controller_runs_with_gurobi():
    ...
```

## Linting

We use Ruff for linting:

```bash
ruff check .
```

Recommended: run Ruff before committing to catch issues early.

## Type hints

The project ships a PEP 561 marker (`py.typed`). New or modified code should:
- Add type annotations where reasonable.
- Avoid `Any` unless necessary; prefer precise types.
- Use `from __future__ import annotations` for forward references when helpful.

If you add a dedicated type-checker (e.g., mypy or pyright) to the dev toolchain, ensure configuration lives in `pyproject.toml` and update this guide.

## Logging and prints

- Do not use `print()` in library code. Use the standard logging library:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  logger.debug("message")
  ```
- Do not configure logging globally in the library. Allow applications to configure logging.

## Building

Build source and wheel distributions using:

```bash
python -m build
# or
hatch build
```

Inspect artifacts in `dist/`. The sdist excludes large datasets/notebooks to keep distributions lean (see pyproject.toml).

## Releases

- Ensure `__version__` (from package metadata) is updated via `pyproject.toml` before tagging a release.
- Build and test the distributions locally.
- Publish using your chosen tooling (e.g., `twine upload dist/*`).

## Coding style

- Follow PEP 8 where practical. Line length is configured in Ruff (120).
- Prefer pure functions and small classes with clear responsibilities.
- Keep optional dependencies optional: import inside functions/methods and provide clear error messages.
- Add unit tests for new functionality and significant bug fixes.

## Questions

Please open a discussion or an issue if anything in this guide is unclear.
