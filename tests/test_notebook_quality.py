import sys

import notebook_quality


def test_discover_notebooks_scope_filter(tmp_path, md_cell, code_cell, write_kernel_bundle):
    write_kernel_bundle(
        tmp_path,
        "competition-a",
        "main.ipynb",
        [md_cell("# Main"), code_cell("print('ok')")],
    )
    write_kernel_bundle(
        tmp_path,
        "datasets/sample",
        "explore.ipynb",
        [md_cell("# Explore"), code_cell("print('ok')")],
    )

    all_notebooks, warnings = notebook_quality.discover_notebooks(tmp_path, scope="all")
    portfolio_notebooks, _ = notebook_quality.discover_notebooks(tmp_path, scope="portfolio")

    assert len(all_notebooks) == 2
    assert len(portfolio_notebooks) == 1
    assert warnings == []


def test_score_notebook_high_quality(tmp_path, md_cell, code_cell, write_kernel_bundle):
    cells = [
        md_cell("# Robust Notebook"),
        md_cell("## Objective\nDefine the goal."),
        md_cell("## Dataset\nData overview and EDA."),
        code_cell("import numpy as np\nnp.random.seed(42)"),
        md_cell("## Method\nModel approach and training pipeline."),
        code_cell("import matplotlib.pyplot as plt\nplt.plot([1,2,3])"),
        md_cell("## Evaluation\nResults and validation metrics."),
        md_cell("### Insight\nObservation: because we regularized, performance improved."),
        md_cell("## Conclusion\nSummary and next steps to improve leaderboard rank."),
    ]
    write_kernel_bundle(tmp_path, "portfolio-x", "guide.ipynb", cells)
    notebook_path = tmp_path / "portfolio-x" / "guide.ipynb"

    score = notebook_quality.score_notebook(notebook_path, tmp_path, min_score=80)

    assert score.score >= 80
    assert score.passed


def test_score_notebook_low_quality(tmp_path, code_cell, write_kernel_bundle):
    cells = [
        code_cell("x = 1"),
        code_cell("print(x)"),
        code_cell("for i in range(3):\n    print(i)"),
    ]
    write_kernel_bundle(tmp_path, "portfolio-y", "short.ipynb", cells)
    notebook_path = tmp_path / "portfolio-y" / "short.ipynb"

    score = notebook_quality.score_notebook(notebook_path, tmp_path, min_score=70)

    assert score.score < 40
    assert not score.passed
    assert any("H1 title" in hint for hint in score.missing)


def test_main_quality_gate_fails(tmp_path, monkeypatch, code_cell, write_kernel_bundle):
    write_kernel_bundle(
        tmp_path,
        "portfolio-z",
        "weak.ipynb",
        [code_cell("print('minimal')"), code_cell("print('minimal')")],
    )
    output_root = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "notebook_quality.py",
            "--root",
            str(tmp_path),
            "--output-root",
            str(output_root),
            "--today",
            "2026-02-22",
            "--min-score",
            "80",
            "--fail-under-threshold",
        ],
    )

    exit_code = notebook_quality.main()

    assert exit_code == 1
    assert (output_root / "reports" / "latest-notebook-quality.md").exists()
    assert (output_root / "reports" / "latest-notebook-quality.json").exists()
    assert (output_root / "reports" / "latest-notebook-quality-fixes.md").exists()
    assert (output_root / "reports" / "latest-notebook-quality-fixes.json").exists()


def test_build_priority_actions_sorted_by_impact(tmp_path, code_cell, write_kernel_bundle):
    cells = [
        code_cell("print('minimal')"),
        code_cell("print('minimal')"),
        code_cell("print('minimal')"),
    ]
    write_kernel_bundle(tmp_path, "portfolio-q", "minimal.ipynb", cells)
    notebook_path = tmp_path / "portfolio-q" / "minimal.ipynb"

    score = notebook_quality.score_notebook(notebook_path, tmp_path, min_score=70)
    actions = notebook_quality.build_priority_actions(score, top_n=3)

    assert actions
    assert len(actions) == 3
    assert actions[0]["impact"] >= actions[1]["impact"]
    assert actions[1]["impact"] >= actions[2]["impact"]


def test_generate_fixer_markdown_contains_checklist(tmp_path, code_cell, write_kernel_bundle):
    low_cells = [code_cell("print('minimal')"), code_cell("print('minimal')")]
    write_kernel_bundle(tmp_path, "portfolio-low", "low.ipynb", low_cells)
    low_notebook = tmp_path / "portfolio-low" / "low.ipynb"
    low_score = notebook_quality.score_notebook(low_notebook, tmp_path, min_score=70)

    markdown = notebook_quality.generate_fixer_markdown(
        [low_score],
        today=notebook_quality.resolve_today("2026-02-22"),
        target_score=85,
        top_actions=3,
        max_notebooks=5,
    )

    assert "Notebook Quality Fix Checklist" in markdown
    assert "`portfolio-low/low.ipynb`" in markdown
    assert "- [ ]" in markdown
