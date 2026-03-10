"""Shared test fixtures for the kaggle repo test suite."""
import json
import sys
from pathlib import Path

import pytest

# Add the repo root to sys.path at import time so package imports work before
# any fixture has a chance to run.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root directory."""
    return ROOT


@pytest.fixture
def md_cell():
    """Return a callable that creates a minimal markdown cell dict."""

    def _md(text: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": text}

    return _md


@pytest.fixture
def code_cell():
    """Return a callable that creates a minimal code cell dict."""

    def _code(text: str) -> dict:
        return {"cell_type": "code", "metadata": {}, "source": text}

    return _code


@pytest.fixture
def write_notebook():
    """Return a callable that writes a minimal valid .ipynb file."""

    def _write_notebook(path: Path, cells: list[dict]) -> None:
        payload = {
            "cells": cells,
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    return _write_notebook


@pytest.fixture
def write_kernel_bundle():
    """Return a callable that creates a directory with kernel-metadata.json and a notebook."""

    def _write_kernel_bundle(
        root: Path, directory: str, code_file: str, cells: list[dict]
    ) -> None:
        target_dir = root / directory
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "kernel-metadata.json").write_text(
            json.dumps({"id": "test/kernel", "code_file": code_file}),
            encoding="utf-8",
        )
        notebook_payload = {
            "cells": cells,
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (target_dir / code_file).write_text(
            json.dumps(notebook_payload), encoding="utf-8"
        )

    return _write_kernel_bundle


@pytest.fixture
def write_queue_json():
    """Return a callable that writes a JSON file containing a list of entries."""

    def _write_queue_json(path: Path, entries: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries), encoding="utf-8")

    return _write_queue_json
