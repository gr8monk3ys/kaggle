from pathlib import Path

from kaggle_portfolio.datasets import dataset_optimizer
from kaggle_portfolio.shared import kaggle_utils


def test_kaggle_command_falls_back_to_module_cli(monkeypatch):
    monkeypatch.setattr(kaggle_utils, "kaggle_cli_path", lambda: None)
    monkeypatch.setattr(
        kaggle_utils.importlib.util,
        "find_spec",
        lambda name: object() if name == "kaggle.cli" else None,
    )
    cmd = kaggle_utils.kaggle_command()
    assert cmd[1:] == ["-m", "kaggle.cli"]


def test_optimize_dataset_fails_on_invalid_metadata_json(tmp_path):
    ds_dir = tmp_path / "broken-meta"
    ds_dir.mkdir(parents=True)
    (ds_dir / "dataset-metadata.json").write_text("{bad json", encoding="utf-8")

    ok = dataset_optimizer.optimize_dataset(ds_dir, push=False)

    assert ok is False


def test_summarize_subprocess_error_prefers_last_meaningful_line():
    message = dataset_optimizer.summarize_subprocess_error(
        "warning one\nwarning two\n",
        "401 Client Error: Unauthorized for url: https://www.kaggle.com/api/v1/blobs/upload\n"
        "  warnings.warn(\n",
    )
    assert message.startswith("401 Client Error: Unauthorized")


def test_analyze_csv_error_contains_file_name(tmp_path):
    bad_csv = tmp_path / "broken.csv"
    bad_csv.write_text("", encoding="utf-8")

    analysis = dataset_optimizer.analyze_csv(bad_csv)

    assert analysis["file"] == "broken.csv"
    assert "error" in analysis


def test_generate_readme_handles_error_entries_without_file_key(tmp_path):
    content = dataset_optimizer.generate_readme(
        ds_dir=tmp_path / "dataset",
        meta={"title": "Sample Dataset"},
        file_analyses=[{"error": "boom", "columns": [], "rows": 0, "size_kb": 0.0}],
    )

    assert "unknown-file" in content
    assert "Error reading file: boom" in content


def test_optimize_dataset_includes_parquet_analysis(tmp_path, monkeypatch):
    ds_dir = tmp_path / "sample-dataset"
    ds_dir.mkdir(parents=True)
    (ds_dir / "dataset-metadata.json").write_text(
        '{"title":"Sample","description":"d","licenses":[{"name":"CC0"}]}',
        encoding="utf-8",
    )
    (ds_dir / "events.parquet").write_bytes(b"PAR1")

    calls: list[str] = []

    def fake_analyze_parquet(path: Path, max_rows: int = 5000) -> dict:
        calls.append(path.name)
        return {
            "file": path.name,
            "rows": 10,
            "size_kb": 1.0,
            "columns": [
                {
                    "name": "event_id",
                    "dtype": "integer",
                    "null_pct": 0.0,
                    "n_unique": 10,
                    "samples": ["1", "2", "3"],
                    "total": 10,
                }
            ],
        }

    monkeypatch.setattr(dataset_optimizer, "analyze_parquet", fake_analyze_parquet)

    ok = dataset_optimizer.optimize_dataset(ds_dir, push=False)
    readme = (ds_dir / "README.md").read_text(encoding="utf-8")

    assert ok
    assert calls == ["events.parquet"]
    assert "events.parquet" in readme


def test_apply_metadata_defaults_sets_update_frequency_when_missing():
    meta = {"id": "owner/sample", "title": "Sample", "description": "desc"}

    normalized, changed = dataset_optimizer.apply_metadata_defaults(meta)

    assert changed is True
    assert normalized["updateFrequency"] == "Monthly"
