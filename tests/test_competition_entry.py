"""Tests for competition_entry.py."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import competition_entry as entry


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------

class TestMakeSlug:
    def test_simple_title(self):
        assert entry.make_slug("Titanic") == "titanic"

    def test_spaces_to_hyphens(self):
        slug = entry.make_slug("My Cool Competition")
        assert slug == "my-cool-competition"

    def test_special_chars_removed(self):
        slug = entry.make_slug("NLP: Disaster Tweets!")
        assert slug == "nlp-disaster-tweets"

    def test_truncation_at_word_boundary(self):
        long_title = "Very Long Competition Title That Exceeds The Maximum Length Limit"
        slug = entry.make_slug(long_title)
        assert len(slug) <= entry.MAX_SLUG_LEN
        assert not slug.endswith("-")

    def test_max_length_enforced(self):
        title = "a" * 100
        slug = entry.make_slug(title)
        assert len(slug) <= entry.MAX_SLUG_LEN

    def test_short_title_unchanged(self):
        slug = entry.make_slug("abc def")
        assert slug == "abc-def"


# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------

class TestDetectCategory:
    def test_nlp_detection(self):
        assert entry.detect_category("NLP Getting Started: Disaster Tweets") == "nlp"

    def test_cv_detection(self):
        assert entry.detect_category("Image Segmentation Challenge") == "cv"

    def test_timeseries_detection(self):
        assert entry.detect_category("Store Sales Forecasting") == "timeseries"

    def test_tabular_default(self):
        assert entry.detect_category("Some Random Competition") == "tabular"

    def test_tabular_explicit(self):
        assert entry.detect_category("Titanic Classification") == "tabular"


# ---------------------------------------------------------------------------
# Kernel metadata generation
# ---------------------------------------------------------------------------

class TestMakeKernelMetadata:
    def test_required_fields_present(self):
        meta = entry.make_kernel_metadata("spaceship-titanic", "Spaceship Titanic: EDA", False)
        assert "id" in meta
        assert "title" in meta
        assert "code_file" in meta
        assert "competition_sources" in meta
        assert "spaceship-titanic" in meta["competition_sources"]

    def test_gpu_flag(self):
        meta_gpu = entry.make_kernel_metadata("test", "Test", True)
        meta_cpu = entry.make_kernel_metadata("test", "Test", False)
        assert meta_gpu["enable_gpu"] is True
        assert meta_cpu["enable_gpu"] is False

    def test_slug_in_id(self):
        meta = entry.make_kernel_metadata("test-comp", "Test Competition: EDA & Baseline", False)
        assert meta["id"].startswith("lorenzoscaturchio/")
        slug_part = meta["id"].split("/")[1]
        assert len(slug_part) <= entry.MAX_SLUG_LEN


# ---------------------------------------------------------------------------
# Cell generation
# ---------------------------------------------------------------------------

class TestGenerateCells:
    def test_generates_15_plus_cells(self):
        cells = entry._generate_cells("test-comp", "Test Competition", "tabular", False)
        assert len(cells) >= 15

    def test_nlp_has_tfidf(self):
        cells = entry._generate_cells("test", "Test NLP", "nlp", False)
        all_source = "".join(
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in cells
        )
        assert "TfidfVectorizer" in all_source

    def test_cv_has_torch_when_gpu(self):
        cells = entry._generate_cells("test", "Test CV", "cv", True)
        all_source = "".join(
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in cells
        )
        assert "import torch" in all_source

    def test_timeseries_has_timeseriessplit(self):
        cells = entry._generate_cells("test", "Test TS", "timeseries", False)
        all_source = "".join(
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in cells
        )
        assert "TimeSeriesSplit" in all_source

    def test_all_cells_have_valid_type(self):
        for cat in ["tabular", "nlp", "cv", "timeseries"]:
            cells = entry._generate_cells("test", "Test", cat, False)
            for cell in cells:
                assert cell["cell_type"] in ("markdown", "code")


# ---------------------------------------------------------------------------
# End-to-end directory creation
# ---------------------------------------------------------------------------

class TestCreateEntry:
    @patch("competition_entry.fetch_competition_info")
    def test_creates_directory(self, mock_fetch, tmp_path):
        mock_fetch.return_value = None

        with patch.object(entry, "ROOT", tmp_path):
            ok = entry.create_entry("test-competition", gpu=False, push=False)

        assert ok
        entry_dir = tmp_path / "test-competition"
        assert entry_dir.is_dir()

        # kernel-metadata.json exists and is valid
        meta_path = entry_dir / "kernel-metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "test-competition" in meta["competition_sources"]

        # Notebook exists
        code_file = meta["code_file"]
        nb_path = entry_dir / code_file
        assert nb_path.exists()
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) >= 15

    @patch("competition_entry.fetch_competition_info")
    def test_existing_directory_updates_notebook(self, mock_fetch, tmp_path):
        mock_fetch.return_value = None

        entry_dir = tmp_path / "existing-comp"
        entry_dir.mkdir()
        # Pre-existing kernel-metadata.json
        meta = {
            "id": "lorenzoscaturchio/existing-comp",
            "title": "Existing",
            "code_file": "notebook.ipynb",
            "competition_sources": ["existing-comp"],
        }
        (entry_dir / "kernel-metadata.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

        with patch.object(entry, "ROOT", tmp_path):
            ok = entry.create_entry("existing-comp", gpu=False, push=False)

        assert ok
        # Should not overwrite existing kernel-metadata.json
        reloaded = json.loads((entry_dir / "kernel-metadata.json").read_text())
        assert reloaded["code_file"] == "notebook.ipynb"

        # But notebook should be created
        assert (entry_dir / "notebook.ipynb").exists()

    @patch("competition_entry.fetch_competition_info")
    def test_fetched_title_used(self, mock_fetch, tmp_path):
        mock_fetch.return_value = {"title": "Amazing Competition", "ref": "amazing-comp"}

        with patch.object(entry, "ROOT", tmp_path):
            ok = entry.create_entry("amazing-comp", gpu=False, push=False)

        assert ok
        meta = json.loads(
            (tmp_path / "amazing-comp" / "kernel-metadata.json").read_text()
        )
        assert "Amazing Competition" in meta["title"]
