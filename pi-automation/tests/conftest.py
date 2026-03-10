"""Shared test fixtures for pi-automation tests."""
import sys
from pathlib import Path

import pytest

# Add pi-automation/scripts and the repo root to sys.path at import time so
# module-level imports work before fixtures run.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "pi-automation" / "scripts"
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def pi_scripts_path() -> Path:
    """Return the path to pi-automation/scripts (already on sys.path)."""
    return SCRIPTS_DIR


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root directory."""
    return REPO_ROOT
