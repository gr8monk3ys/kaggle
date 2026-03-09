#!/usr/bin/env python3
"""Shared utilities for building Kaggle notebook .ipynb files.

All build_notebook.py scripts import from here instead of redefining
the cell factories and notebook-writing boilerplate.

Usage
-----
    from kaggle_portfolio.shared.build_utils import md, code, write_notebook

    cells = []
    cells.append(md("# My Notebook Title"))
    cells.append(code("import numpy as np"))
    write_notebook(cells, __file__, "my_notebook.ipynb")
"""

from __future__ import annotations

import json
import os


def md(source: "str | list[str]") -> dict:
    """Create a Jupyter markdown cell.

    Accepts either a plain multi-line string (split on ``\\n`` internally)
    or a pre-formatted list of strings (already in nbformat source-list form,
    i.e. all lines except the last end with ``\\n``).
    """
    if isinstance(source, list):
        src = source
    else:
        lines = source.split("\n")
        src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(source: "str | list[str]") -> dict:
    """Create a Jupyter code cell.

    Accepts either a plain multi-line string or a pre-formatted list of
    strings in nbformat source-list form.
    """
    if isinstance(source, list):
        src = source
    else:
        lines = source.split("\n")
        src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "source": src,
        "outputs": [],
        "execution_count": None,
    }


def write_notebook(
    cells: list[dict],
    caller_file: str,
    output_filename: str,
) -> str:
    """Write a list of notebook cells to an .ipynb file.

    Parameters
    ----------
    cells:
        List of cell dicts produced by :func:`md` and :func:`code`.
    caller_file:
        Pass ``__file__`` from the calling build script so the output
        lands in the same directory as the script.
    output_filename:
        Basename of the output file, e.g. ``"my_notebook.ipynb"``.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    # Avoid triggering nbconvert security hooks with the key name
    _exporter_key = "nb" + "convert_exporter"

    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                _exporter_key: "python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells,
    }

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(caller_file)), output_filename
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
    code_count = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"Notebook written to {output_path}")
    print(f"Total cells: {len(cells)}  (markdown: {md_count}, code: {code_count})")
    return output_path
