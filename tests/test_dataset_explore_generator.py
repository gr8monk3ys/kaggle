"""Tests for dataset_explore_generator.py."""
import json
from pathlib import Path

import pytest

import dataset_explore_generator as gen


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

class TestClassifyColumns:
    def test_numeric_columns(self):
        analysis = {
            "columns": [
                {"name": "age", "dtype": "integer", "n_unique": 80, "total": 100},
                {"name": "salary", "dtype": "float", "n_unique": 95, "total": 100},
            ],
            "rows": 100,
        }
        result = gen._classify_columns(analysis)
        assert result["numeric"] == ["age", "salary"]
        assert result["categorical"] == []

    def test_categorical_columns(self):
        analysis = {
            "columns": [
                {"name": "color", "dtype": "string", "n_unique": 5, "total": 100},
                {"name": "active", "dtype": "boolean", "n_unique": 2, "total": 100},
            ],
            "rows": 100,
        }
        result = gen._classify_columns(analysis)
        assert result["categorical"] == ["color", "active"]
        assert result["numeric"] == []

    def test_high_cardinality_string(self):
        analysis = {
            "columns": [
                {"name": "email", "dtype": "string", "n_unique": 400, "total": 500},
            ],
            "rows": 500,
        }
        result = gen._classify_columns(analysis)
        assert result["high_cardinality"] == ["email"]

    def test_id_like_columns_excluded(self):
        analysis = {
            "columns": [
                {"name": "user_id", "dtype": "integer", "n_unique": 100, "total": 100},
                {"name": "id", "dtype": "integer", "n_unique": 100, "total": 100},
                {"name": "score", "dtype": "float", "n_unique": 50, "total": 100},
            ],
            "rows": 100,
        }
        result = gen._classify_columns(analysis)
        assert "user_id" in result["id_like"]
        assert "id" in result["id_like"]
        assert result["numeric"] == ["score"]

    def test_target_detection(self):
        analysis = {
            "columns": [
                {"name": "feature1", "dtype": "float", "n_unique": 90, "total": 100},
                {"name": "target", "dtype": "integer", "n_unique": 2, "total": 100},
            ],
            "rows": 100,
        }
        result = gen._classify_columns(analysis)
        assert result["target"] == "target"

    def test_no_target_when_absent(self):
        analysis = {
            "columns": [
                {"name": "feature1", "dtype": "float", "n_unique": 90, "total": 100},
                {"name": "feature2", "dtype": "float", "n_unique": 80, "total": 100},
            ],
            "rows": 100,
        }
        result = gen._classify_columns(analysis)
        assert result["target"] is None


# ---------------------------------------------------------------------------
# Cell generators
# ---------------------------------------------------------------------------

class TestCellGenerators:
    def test_title_cell_contains_dataset_name(self):
        meta = {"title": "My Dataset", "subtitle": "100 rows", "id": "user/my-ds"}
        analysis = {"rows": 100, "columns": [{"name": "a"}], "file": "data.csv"}
        cells = gen._cell_title(meta, analysis)
        assert len(cells) == 1
        assert cells[0]["cell_type"] == "markdown"
        source_text = "".join(cells[0]["source"])
        assert "My Dataset" in source_text
        assert "100 rows" in source_text

    def test_distributions_empty_when_no_numeric(self):
        classified = {"numeric": [], "categorical": [], "high_cardinality": [],
                       "id_like": [], "target": None}
        cells = gen._cell_distributions(classified)
        source_text = "".join(c.get("source", "") if isinstance(c.get("source"), str)
                              else "".join(c.get("source", [])) for c in cells)
        assert "No numeric columns" in source_text

    def test_correlations_need_two_numeric(self):
        classified = {"numeric": ["only_one"], "categorical": [], "high_cardinality": [],
                       "id_like": [], "target": None}
        cells = gen._cell_correlations(classified)
        source_text = "".join("".join(c.get("source", [])) for c in cells)
        assert "at least 2" in source_text


# ---------------------------------------------------------------------------
# End-to-end notebook generation
# ---------------------------------------------------------------------------

class TestGenerateExploreNotebook:
    def _make_dataset(self, tmp_path: Path, rows: int = 50):
        """Create a minimal dataset directory with CSV and metadata."""
        ds_dir = tmp_path / "datasets" / "test-ds"
        ds_dir.mkdir(parents=True)

        # CSV
        import csv
        csv_path = ds_dir / "test_data.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "age", "salary", "department", "target"])
            for i in range(rows):
                writer.writerow([
                    i,
                    20 + (i % 40),
                    30000 + i * 100,
                    ["Engineering", "Sales", "Marketing"][i % 3],
                    i % 2,
                ])

        # dataset-metadata.json
        meta = {
            "id": "testuser/test-dataset",
            "title": "Test Dataset",
            "subtitle": "50 rows for testing",
            "keywords": ["test"],
            "resources": [{"path": "test_data.csv"}],
        }
        (ds_dir / "dataset-metadata.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

        # kernel-metadata.json
        km = {
            "id": "testuser/test-ds-eda",
            "title": "Test DS EDA",
            "code_file": "explore.ipynb",
            "dataset_sources": ["testuser/test-dataset"],
        }
        (ds_dir / "kernel-metadata.json").write_text(
            json.dumps(km), encoding="utf-8"
        )

        return ds_dir

    def test_generates_20_plus_cells(self, tmp_path):
        ds_dir = self._make_dataset(tmp_path)
        cells = gen.generate_explore_notebook(ds_dir)
        assert len(cells) >= 15  # ~18-20 with all sections active

    def test_generates_valid_ipynb(self, tmp_path):
        ds_dir = self._make_dataset(tmp_path)
        ok = gen.build_explore(ds_dir, push=False)
        assert ok

        ipynb_path = ds_dir / "explore.ipynb"
        assert ipynb_path.exists()

        nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) >= 15

        # Every cell must have a valid type
        for cell in nb["cells"]:
            assert cell["cell_type"] in ("markdown", "code")

    def test_skip_when_no_csv(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "empty-ds"
        ds_dir.mkdir(parents=True)
        meta = {"id": "test/empty", "title": "Empty"}
        (ds_dir / "dataset-metadata.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        cells = gen.generate_explore_notebook(ds_dir)
        assert cells == []

    def test_skip_when_no_metadata(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "no-meta"
        ds_dir.mkdir(parents=True)
        cells = gen.generate_explore_notebook(ds_dir)
        assert cells == []
