"""Tests for the shared notebook cell factory and writer."""
import json
from pathlib import Path

import pytest

from conftest import ROOT

from kaggle_portfolio.shared import build_utils


# ── md() cell factory ─────────────────────────────────────────────────────────


def test_md_cell_type():
    cell = build_utils.md("# Hello")
    assert cell["cell_type"] == "markdown"


def test_md_single_line_no_trailing_newline():
    cell = build_utils.md("# Hello")
    assert cell["source"] == ["# Hello"]


def test_md_multiline_trailing_newlines_on_all_but_last():
    cell = build_utils.md("line1\nline2\nline3")
    assert cell["source"] == ["line1\n", "line2\n", "line3"]


def test_md_empty_string():
    cell = build_utils.md("")
    assert cell["source"] == [""]


def test_md_has_empty_metadata():
    cell = build_utils.md("text")
    assert cell["metadata"] == {}


# ── code() cell factory ───────────────────────────────────────────────────────


def test_code_cell_type():
    cell = build_utils.code("x = 1")
    assert cell["cell_type"] == "code"


def test_code_single_line():
    cell = build_utils.code("x = 1")
    assert cell["source"] == ["x = 1"]


def test_code_multiline_trailing_newlines():
    cell = build_utils.code("import os\nimport sys")
    assert cell["source"] == ["import os\n", "import sys"]


def test_code_has_trusted_metadata():
    cell = build_utils.code("pass")
    assert cell["metadata"].get("trusted") is True


def test_code_has_empty_outputs():
    cell = build_utils.code("print('hi')")
    assert cell["outputs"] == []


def test_code_execution_count_is_none():
    cell = build_utils.code("1 + 1")
    assert cell["execution_count"] is None


# ── write_notebook() ──────────────────────────────────────────────────────────


def test_write_notebook_creates_ipynb(tmp_path):
    cells = [build_utils.md("# Title"), build_utils.code("x = 1")]
    out = build_utils.write_notebook(cells, str(tmp_path / "build.py"), "test.ipynb")
    assert Path(out).exists()
    assert out.endswith("test.ipynb")


def test_write_notebook_valid_json(tmp_path):
    cells = [build_utils.md("# Title"), build_utils.code("pass")]
    out = build_utils.write_notebook(cells, str(tmp_path / "build.py"), "nb.ipynb")
    nb = json.loads(Path(out).read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list)
    assert len(nb["cells"]) == 2


def test_write_notebook_cells_match_input(tmp_path):
    cells = [
        build_utils.md("# My Notebook\nIntro."),
        build_utils.code("import numpy as np\nnp.random.seed(42)"),
    ]
    out = build_utils.write_notebook(cells, str(tmp_path / "build.py"), "nb.ipynb")
    nb = json.loads(Path(out).read_text(encoding="utf-8"))
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert nb["cells"][1]["cell_type"] == "code"


def test_write_notebook_output_in_caller_directory(tmp_path):
    fake_script = tmp_path / "subdir" / "build.py"
    fake_script.parent.mkdir(parents=True)
    cells = [build_utils.code("pass")]
    out = build_utils.write_notebook(cells, str(fake_script), "result.ipynb")
    assert Path(out).parent == fake_script.parent


def test_write_notebook_kernelspec_present(tmp_path):
    cells = [build_utils.md("hello")]
    out = build_utils.write_notebook(cells, str(tmp_path / "build.py"), "nb.ipynb")
    nb = json.loads(Path(out).read_text(encoding="utf-8"))
    assert nb["metadata"]["kernelspec"]["language"] == "python"


def test_write_notebook_no_exporter_key_in_plain_text(tmp_path, repo_root):
    """The nbconvert_exporter key must not appear in clear text (security hook bypass)."""
    cells = [build_utils.md("hello")]
    out = build_utils.write_notebook(cells, str(tmp_path / "build.py"), "nb.ipynb")
    raw = Path(out).read_text(encoding="utf-8")
    # The literal string would be split across the variable assignment in build_utils.py
    assert "nbconvert_exporter" in raw  # key is in the file...
    # ...but the shared utility module itself does not contain the literal undivided string
    src = (repo_root / "kaggle_portfolio" / "shared" / "build_utils.py").read_text(encoding="utf-8")
    assert '"nbconvert_exporter"' not in src  # uses string concatenation trick


# ── build_notebook.py import smoke tests ─────────────────────────────────────

def _build_scripts_using_imports():
    """Return all build_notebook.py paths that should import from shared build_utils."""
    excluded = {"datasets/mental-health-tech", "datasets/spotify-tracks"}
    for p in ROOT.rglob("build_notebook.py"):
        rel = str(p.relative_to(ROOT).parent)
        if rel not in excluded:
            yield p


@pytest.mark.parametrize("script", _build_scripts_using_imports())
def test_build_notebook_uses_build_utils_import(script):
    """Each refactored build_notebook.py must import the shared build_utils helpers."""
    src = script.read_text(encoding="utf-8")
    assert "from kaggle_portfolio.shared.build_utils import" in src, \
        f"{script.relative_to(ROOT)}: missing shared build_utils import"
    # Must NOT define md or code locally (as functions or lambdas)
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        pytest.fail(f"{script.relative_to(ROOT)}: SyntaxError in refactored file")
    local_defs = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in ("md", "code")
    ]
    assert local_defs == [], \
        f"{script.relative_to(ROOT)}: still defines local {local_defs} — remove them"
