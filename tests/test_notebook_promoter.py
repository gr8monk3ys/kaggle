import json
from pathlib import Path

from kaggle_portfolio.notebooks import notebook_promoter


def _write_meta(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_notebooks_reports_metadata_warnings(tmp_path, monkeypatch):
    monkeypatch.setattr(notebook_promoter, "ROOT", tmp_path)

    _write_meta(
        tmp_path / "good-notebook" / "kernel-metadata.json",
        {"id": "user/good", "title": "Good Notebook"},
    )

    bad_dir = tmp_path / "bad-json"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "kernel-metadata.json").write_text("{not json", encoding="utf-8")

    _write_meta(
        tmp_path / "missing-id" / "kernel-metadata.json",
        {"title": "Missing Id"},
    )

    notebooks, warnings = notebook_promoter.load_notebooks()

    assert len(notebooks) == 1
    assert notebooks[0]["id"] == "user/good"
    assert len(warnings) == 2
    assert any("bad-json/kernel-metadata.json" in warning for warning in warnings)
    assert any("missing-id/kernel-metadata.json" in warning for warning in warnings)


def test_main_strict_metadata_fails_when_warnings_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(notebook_promoter, "ROOT", tmp_path)
    _write_meta(
        tmp_path / "missing-id" / "kernel-metadata.json",
        {"title": "Missing Id"},
    )

    rc = notebook_promoter.main(["--strict-metadata"])
    assert rc == 1


def test_load_notebooks_skips_dataset_explorer_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(notebook_promoter, "ROOT", tmp_path)
    _write_meta(
        tmp_path / "datasets" / "example-dataset" / "kernel-metadata.json",
        {"id": "user/dataset-explorer", "title": "Dataset Explorer"},
    )
    _write_meta(
        tmp_path / "competition-notebook" / "kernel-metadata.json",
        {"id": "user/competition-notebook", "title": "Competition Notebook"},
    )

    notebooks, warnings = notebook_promoter.load_notebooks()

    assert warnings == []
    assert [item["id"] for item in notebooks] == ["user/competition-notebook"]


def test_match_notebook_to_competitions_uses_keywords_and_requires_overlap():
    notebook = {
        "id": "user/store-sales-notebook",
        "title": "Store Sales Time Series Forecasting",
        "tags": [],
        "keywords": ["lightgbm", "lag features"],
    }

    matches = notebook_promoter.match_notebook_to_competitions(notebook)

    assert "store-sales-time-series-forecasting" in matches


def test_filter_notebooks_matches_ref_slug_and_directory():
    notebooks = [
        {"id": "user/one", "_dir": "dir-one"},
        {"id": "user/two", "_dir": "dir-two"},
    ]

    filtered, missing = notebook_promoter.filter_notebooks(notebooks, {"user/two", "dir-one", "missing"})

    assert [item["id"] for item in filtered] == ["user/one", "user/two"]
    assert missing == ["missing"]
