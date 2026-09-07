from kaggle_portfolio.notebooks import notebook_pipeline


def test_score_notebook_dir_returns_non_zero_for_valid_notebook(
    tmp_path, md_cell, code_cell, write_notebook
):
    nb_dir = tmp_path / "sample"
    nb_dir.mkdir(parents=True)
    write_notebook(
        nb_dir / "guide.ipynb",
        [
            md_cell("# Quality Notebook"),
            md_cell("## Objective\nDefine task and metric."),
            md_cell("## Dataset\nQuick EDA summary."),
            code_cell("import numpy as np\nnp.random.seed(42)"),
            md_cell("## Method\nTraining pipeline."),
            code_cell("import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])"),
            md_cell("## Evaluation\nValidation results."),
            md_cell("## Conclusion\nNext steps and improvements."),
        ],
    )

    score, summary = notebook_pipeline.score_notebook_dir(nb_dir)

    assert score > 0
    assert "Score:" in summary


def test_discover_build_scripts_includes_underscore_variant(repo_root):
    scripts = notebook_pipeline.discover_build_scripts(repo_root)
    assert (
        repo_root
        / "projects"
        / "educational"
        / "timeseries-transformers"
        / "_build_notebook.py"
    ) in scripts
