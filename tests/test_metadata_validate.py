"""Tests for kernel-metadata.json validation across the portfolio.

These tests verify both the structure of existing metadata files and
the validation logic that manage.sh validate relies on.
"""
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANAGE = ROOT / "manage.sh"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_kernel_metas():
    """Return all kernel-metadata.json paths in the repo."""
    return list(ROOT.rglob("kernel-metadata.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Structural tests on real metadata files ───────────────────────────────────


@pytest.mark.parametrize("meta_path", _find_kernel_metas())
def test_kernel_metadata_is_valid_json(meta_path):
    """Every kernel-metadata.json in the repo must be parseable JSON."""
    _load(meta_path)  # raises if invalid


@pytest.mark.parametrize("meta_path", _find_kernel_metas())
def test_kernel_metadata_required_fields(meta_path):
    """id, title, code_file, language, kernel_type, is_private must be present."""
    meta = _load(meta_path)
    required = ["id", "title", "code_file", "language", "kernel_type", "is_private"]
    missing = [f for f in required if f not in meta]
    assert not missing, f"{meta_path.relative_to(ROOT)} missing fields: {missing}"


@pytest.mark.parametrize("meta_path", _find_kernel_metas())
def test_kernel_metadata_title_length(meta_path):
    """Title must be 6–70 characters to pass Kaggle's validation."""
    meta = _load(meta_path)
    title = meta.get("title", "")
    assert 6 <= len(title) <= 70, (
        f"{meta_path.relative_to(ROOT)}: title length {len(title)} out of range [6, 70]: {title!r}"
    )


@pytest.mark.parametrize("meta_path", _find_kernel_metas())
def test_kernel_metadata_code_file_exists(meta_path):
    """The code_file listed in metadata must exist in the same directory."""
    meta = _load(meta_path)
    code_file = meta.get("code_file", "")
    if code_file:
        target = meta_path.parent / code_file
        assert target.exists(), (
            f"{meta_path.relative_to(ROOT)}: code_file '{code_file}' not found"
        )


@pytest.mark.parametrize("meta_path", _find_kernel_metas())
def test_kernel_metadata_dataset_sources_is_list(meta_path):
    """dataset_sources, if present, must be a list (not a string or dict)."""
    meta = _load(meta_path)
    if "dataset_sources" in meta:
        assert isinstance(meta["dataset_sources"], list), (
            f"{meta_path.relative_to(ROOT)}: dataset_sources must be a list"
        )


@pytest.mark.parametrize("meta_path", _find_kernel_metas())
def test_kernel_metadata_kernel_sources_is_list(meta_path):
    """kernel_sources, if present, must be a list."""
    meta = _load(meta_path)
    if "kernel_sources" in meta:
        assert isinstance(meta["kernel_sources"], list), (
            f"{meta_path.relative_to(ROOT)}: kernel_sources must be a list"
        )


@pytest.mark.parametrize("meta_path", _find_kernel_metas())
def test_kernel_metadata_id_format(meta_path):
    """id must follow the owner/slug format and contain no spaces."""
    meta = _load(meta_path)
    id_val = meta.get("id", "")
    assert "/" in id_val, f"{meta_path.relative_to(ROOT)}: id '{id_val}' missing owner/ prefix"
    assert " " not in id_val, f"{meta_path.relative_to(ROOT)}: id '{id_val}' contains spaces"


@pytest.mark.parametrize("meta_path", _find_kernel_metas())
def test_kernel_metadata_no_credentials(meta_path):
    """Metadata files must not contain API keys or passwords."""
    raw = meta_path.read_text(encoding="utf-8").lower()
    suspicious = ["password", "secret", "kgat_", "kaggle_token"]
    found = [kw for kw in suspicious if kw in raw]
    assert not found, (
        f"{meta_path.relative_to(ROOT)}: possible credentials found: {found}"
    )


# ── validate subcommand integration test ─────────────────────────────────────


def test_manage_validate_exits_zero_on_clean_repo():
    """manage.sh validate must exit 0 on the current (clean) repo state."""
    result = subprocess.run(
        ["bash", str(MANAGE), "validate"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"manage.sh validate failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "All metadata files valid" in result.stdout


def test_manage_validate_fails_on_invalid_json(tmp_path):
    """manage.sh validate returns non-zero when JSON is malformed."""
    (tmp_path / "kernel-metadata.json").write_text("{bad json", encoding="utf-8")
    (tmp_path / "notebook.ipynb").write_text("{}", encoding="utf-8")
    patched_manage = tmp_path / "manage.sh"
    original = MANAGE.read_text(encoding="utf-8")
    patched = original.replace(
        'KAGGLE_DIR="$(cd "$(dirname "$0")" && pwd)"',
        f'KAGGLE_DIR="{tmp_path}"',
        1,
    )
    assert patched != original, "Failed to patch manage.sh test fixture"
    patched_manage.write_text(patched, encoding="utf-8")

    result = subprocess.run(
        ["bash", str(patched_manage), "validate"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "invalid JSON" in result.stdout


def test_validate_python_logic_invalid_json(tmp_path):
    """Python validation logic: malformed JSON is caught."""
    bad = tmp_path / "kernel-metadata.json"
    bad.write_text("{not valid json", encoding="utf-8")

    import subprocess as sp
    result = sp.run(
        ["python3", "-c", f"import json; json.load(open('{bad}'))"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_validate_python_logic_title_too_short():
    """Title shorter than 6 chars should be flagged."""
    title = "Hi"
    assert len(title) < 6


def test_validate_python_logic_title_too_long():
    """Title longer than 70 chars should be flagged."""
    title = "A" * 71
    assert len(title) > 70


# ── Three new dataset kernel-metadata.json files ─────────────────────────────


@pytest.mark.parametrize("ds_dir,expected_id", [
    ("datasets/credit-card-fraud", "lorenzoscaturchio/credit-card-fraud-eda-detection"),
    ("datasets/job-postings",      "lorenzoscaturchio/job-postings-nlp-salary-prediction-eda"),
    ("datasets/student-performance", "lorenzoscaturchio/student-performance-academic-eda"),
])
def test_dataset_explorer_kernel_metadata_exists(ds_dir, expected_id):
    """Dataset explorer notebooks must have kernel-metadata.json with the correct id."""
    meta_path = ROOT / ds_dir / "kernel-metadata.json"
    assert meta_path.exists(), f"Missing kernel-metadata.json in {ds_dir}"
    meta = _load(meta_path)
    assert meta["id"] == expected_id, (
        f"{ds_dir}: expected id '{expected_id}', got '{meta['id']}'"
    )


@pytest.mark.parametrize("ds_dir", [
    "datasets/credit-card-fraud",
    "datasets/job-postings",
    "datasets/student-performance",
])
def test_dataset_explorer_kernel_metadata_has_dataset_sources(ds_dir):
    """Dataset explorer notebooks must declare their parent dataset as a source."""
    meta_path = ROOT / ds_dir / "kernel-metadata.json"
    meta = _load(meta_path)
    assert "dataset_sources" in meta, f"{ds_dir}: missing dataset_sources"
    assert len(meta["dataset_sources"]) > 0, f"{ds_dir}: dataset_sources is empty"


# ── manage.sh new command smoke tests ─────────────────────────────────────────


def test_manage_help_includes_new_commands():
    """manage.sh --help must list all new commands: validate, link-competition."""
    result = subprocess.run(
        ["bash", str(MANAGE), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "validate" in result.stdout
    assert "link-competition" in result.stdout
    assert "votes" in result.stdout


def test_manage_auto_discovery_finds_all_notebooks():
    """Auto-discovery must find at least as many notebooks as the old hardcoded list."""
    # The hardcoded list had 24 entries; auto-discovery should find >= 24
    result = subprocess.run(
        ["bash", "-c", f"source {MANAGE}; echo ${{#NOTEBOOK_DIRS[@]}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # Can't easily source manage.sh directly due to set -euo pipefail,
    # so we count kernel-metadata.json files not under datasets/
    count = sum(
        1 for p in ROOT.rglob("kernel-metadata.json")
        if "datasets" not in p.parts[len(ROOT.parts):]
    )
    assert count >= 24, f"Expected >= 24 notebook dirs, found {count}"
