from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import dataset_usability


def _write_dataset_bundle(
    root: Path,
    name: str,
    metadata: dict,
    readme: str | None = None,
    data_files: list[str] | None = None,
    include_create: bool = True,
    include_explore: bool = True,
    include_kernel_meta: bool = True,
) -> Path:
    ds_dir = root / "datasets" / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "dataset-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    if readme is not None:
        (ds_dir / "README.md").write_text(readme, encoding="utf-8")
    if include_create:
        (ds_dir / "create_dataset.py").write_text("print('ok')\n", encoding="utf-8")
    if include_explore:
        (ds_dir / "explore.ipynb").write_text("{}", encoding="utf-8")
    if include_kernel_meta:
        (ds_dir / "kernel-metadata.json").write_text(
            json.dumps({"id": "u/explore", "title": "Explore", "code_file": "explore.ipynb"}),
            encoding="utf-8",
        )
    for filename in data_files or []:
        (ds_dir / filename).write_text("a,b\n1,2\n", encoding="utf-8")
    return ds_dir


def test_score_dataset_high_for_complete_bundle(tmp_path):
    description = "Detailed description. " * 60
    readme = "\n".join(
        [
            "# Dataset",
            "",
            "## Description",
            "Long-form docs.",
            "",
            "## Tags",
            "`ml`, `nlp`, `classification`, `baseline`, `analysis`",
            "",
            "**Kaggle:** [u/sample](https://www.kaggle.com/datasets/u/sample)",
            "",
            "## sample.csv",
            "",
            "| Column | Type | Null% | Unique | Sample values |",
            "|--------|------|-------|--------|---------------|",
            "| `a` | integer | 0.0% | 1 | `1` |",
            "",
            "## Suggested Use Cases",
            "- baseline model",
            "",
            ("filler " * 500).strip(),
        ]
    )
    metadata = {
        "title": "Sample",
        "id": "u/sample",
        "subtitle": "Clean sample",
        "description": description,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["ml", "nlp", "classification", "baseline", "analysis"],
    }
    ds_dir = _write_dataset_bundle(
        tmp_path,
        "sample",
        metadata,
        readme=readme,
        data_files=["sample.csv", "sample.parquet"],
    )

    scored = dataset_usability.score_dataset(ds_dir, root=tmp_path)

    assert scored.score >= 85
    assert scored.tier == "Excellent"
    assert not [issue for issue in scored.issues if "missing" in issue.lower()]


def test_score_dataset_low_when_core_assets_missing(tmp_path):
    metadata = {
        "title": "Weak",
        "id": "",
        "description": "short",
        "licenses": [],
        "keywords": [],
    }
    ds_dir = _write_dataset_bundle(
        tmp_path,
        "weak",
        metadata,
        readme=None,
        data_files=[],
        include_create=False,
        include_explore=False,
        include_kernel_meta=False,
    )

    scored = dataset_usability.score_dataset(ds_dir, root=tmp_path)

    assert scored.score < 55
    assert scored.tier == "At Risk"
    assert any("README.md missing" in issue for issue in scored.issues)


def test_main_strict_returns_nonzero_when_below_gate(tmp_path, monkeypatch):
    metadata = {
        "title": "Weak",
        "id": "u/weak",
        "description": "short",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["ml"],
    }
    _write_dataset_bundle(
        tmp_path,
        "weak",
        metadata,
        readme=None,
        data_files=["weak.csv"],
        include_create=False,
        include_explore=False,
        include_kernel_meta=False,
    )

    out_root = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_usability.py",
            "--root",
            str(tmp_path),
            "--output-root",
            str(out_root),
            "--today",
            "2026-02-24",
            "--fail-under",
            "80",
            "--strict",
        ],
    )

    rc = dataset_usability.main()

    assert rc == 1
    assert (out_root / "reports" / "latest-dataset-usability.md").exists()
    assert (out_root / "reports" / "latest-dataset-usability.json").exists()


def test_generate_markdown_and_json_include_gap_priorities():
    scores = [
        dataset_usability.DatasetScore(
            path="datasets/a",
            title="A",
            score=80,
            score_10=8,
            tier="Good",
            criteria={
                "metadata_core": 20,
                "documentation": 25,
                "data_assets": 15,
                "notebook_assets": 10,
                "discovery": 8,
            },
            issues=["README missing tags section.", "Metadata missing subtitle."],
            data_files=2,
        ),
        dataset_usability.DatasetScore(
            path="datasets/b",
            title="B",
            score=78,
            score_10=8,
            tier="Good",
            criteria={
                "metadata_core": 19,
                "documentation": 24,
                "data_assets": 14,
                "notebook_assets": 10,
                "discovery": 9,
            },
            issues=["README missing tags section."],
            data_files=1,
        ),
    ]

    markdown = dataset_usability.generate_markdown(scores, today=date(2026, 2, 24), fail_under=75)
    payload = dataset_usability.build_json_report(scores, today=date(2026, 2, 24), fail_under=75)

    assert "## Criteria Averages" in markdown
    assert "`metadata_core`" in markdown
    assert "## Common Gaps" in markdown
    assert "Kaggle `usabilityRating` is a separate 0.0-1.0 platform metric." in markdown
    assert "2 dataset(s): README missing tags section." in markdown
    assert payload["summary"]["criteria_average"]["metadata_core"] == 19.5
    assert payload["common_gaps"][0]["issue"] == "README missing tags section."
    assert payload["common_gaps"][0]["count"] == 2
    assert payload["summary"]["average_score_10"] == 8.0


def test_score_out_of_10_ceiling_behavior():
    assert dataset_usability.score_out_of_10(93) == 10
    assert dataset_usability.score_out_of_10(90) == 9
    assert dataset_usability.score_out_of_10(81) == 9
    assert dataset_usability.score_out_of_10(0) == 0


def test_parse_kaggle_datasets_csv_extracts_ratings():
    raw = (
        "ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating\n"
        "u/a,A,1,2026-02-24,0,0,0.64\n"
        "u/b,B,1,2026-02-24,0,0,0.58\n"
    )

    parsed = dataset_usability.parse_kaggle_datasets_csv(raw)

    assert parsed["u/a"] == 0.64
    assert parsed["u/b"] == 0.58


def test_attach_kaggle_live_ratings_sets_optional_fields():
    scores = [
        dataset_usability.DatasetScore(
            path="datasets/a",
            title="A",
            score=90,
            score_10=9,
            tier="Excellent",
            criteria={},
            issues=[],
            data_files=1,
            dataset_ref="u/a",
        ),
        dataset_usability.DatasetScore(
            path="datasets/c",
            title="C",
            score=70,
            score_10=7,
            tier="Good",
            criteria={},
            issues=[],
            data_files=1,
            dataset_ref="u/c",
        ),
    ]

    updated = dataset_usability.attach_kaggle_live_ratings(scores, {"u/a": 0.64})

    assert updated[0].kaggle_usability_rating == 0.64
    assert updated[0].kaggle_score_10 == 6.4
    assert updated[1].kaggle_usability_rating is None


def test_infer_owner_from_scores_uses_majority_owner():
    scores = [
        dataset_usability.DatasetScore(
            path="datasets/a",
            title="A",
            score=90,
            score_10=9,
            tier="Excellent",
            criteria={},
            issues=[],
            data_files=1,
            dataset_ref="owner-one/a",
        ),
        dataset_usability.DatasetScore(
            path="datasets/b",
            title="B",
            score=90,
            score_10=9,
            tier="Excellent",
            criteria={},
            issues=[],
            data_files=1,
            dataset_ref="owner-one/b",
        ),
        dataset_usability.DatasetScore(
            path="datasets/c",
            title="C",
            score=90,
            score_10=9,
            tier="Excellent",
            criteria={},
            issues=[],
            data_files=1,
            dataset_ref="owner-two/c",
        ),
    ]

    assert dataset_usability.infer_owner_from_scores(scores) == "owner-one"


def test_fetch_kaggle_live_ratings_combines_mine_and_search(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "owner")

    def fake_run(args):
        if args == ["--mine", "--csv"]:
            return {"owner/a": 0.61}, None
        if args == ["-s", "owner", "--csv"]:
            return {"owner/b": 0.58}, None
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(dataset_usability, "_run_kaggle_dataset_list", fake_run)

    ratings, err = dataset_usability.fetch_kaggle_live_ratings("owner")

    assert err is None
    assert ratings["owner/a"] == 0.61
    assert ratings["owner/b"] == 0.58


def test_build_live_priority_queue_orders_by_status_then_rating():
    scores = [
        dataset_usability.DatasetScore(
            path="datasets/critical",
            title="Critical",
            score=90,
            score_10=9,
            tier="Excellent",
            criteria={},
            issues=[],
            data_files=1,
            dataset_ref="u/critical",
            kaggle_usability_rating=0.66,
        ),
        dataset_usability.DatasetScore(
            path="datasets/watch",
            title="Watch",
            score=90,
            score_10=9,
            tier="Excellent",
            criteria={},
            issues=[],
            data_files=1,
            dataset_ref="u/watch",
            kaggle_usability_rating=0.74,
        ),
        dataset_usability.DatasetScore(
            path="datasets/strong",
            title="Strong",
            score=90,
            score_10=9,
            tier="Excellent",
            criteria={},
            issues=[],
            data_files=1,
            dataset_ref="u/strong",
            kaggle_usability_rating=0.89,
        ),
    ]

    queue = dataset_usability.build_live_priority_queue(
        scores,
        alert_under=0.7,
        target_rating=0.8,
    )
    summary = dataset_usability.summarize_live_queue(queue)

    assert [item["dataset_ref"] for item in queue] == ["u/critical", "u/watch", "u/strong"]
    assert summary["critical"] == 1
    assert summary["watch"] == 1
    assert summary["strong"] == 1


def test_main_daily_tracker_fails_when_live_alert_gate_triggered(tmp_path, monkeypatch):
    readme = "\n".join(
        [
            "# Dataset",
            "",
            "## Description",
            "Some description",
            "",
            "## Tags",
            "`ml`, `tabular`, `baseline`, `classification`, `analysis`",
            "",
            "**Kaggle:** [u/critical](https://www.kaggle.com/datasets/u/critical)",
            "",
            "## sample.csv",
            "",
            "| Column | Type | Null% | Unique | Sample values |",
            "|--------|------|-------|--------|---------------|",
            "| `a` | integer | 0.0% | 1 | `1` |",
            "",
            "## Suggested Use Cases",
            "- baseline",
            "",
            ("filler " * 300).strip(),
        ]
    )
    meta_common = {
        "title": "Sample",
        "subtitle": "subtitle",
        "description": "Detailed description. " * 40,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["ml", "classification", "analysis", "baseline", "tabular"],
    }
    _write_dataset_bundle(
        tmp_path,
        "critical",
        {**meta_common, "id": "u/critical"},
        readme=readme,
        data_files=["critical.csv"],
    )
    _write_dataset_bundle(
        tmp_path,
        "strong",
        {**meta_common, "id": "u/strong"},
        readme=readme.replace("u/critical", "u/strong"),
        data_files=["strong.csv"],
    )

    live_csv = tmp_path / "live.csv"
    live_csv.write_text(
        "\n".join(
            [
                "ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating",
                "u/critical,Critical,1,2026-02-24,0,0,0.66",
                "u/strong,Strong,1,2026-02-24,0,0,0.85",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_root = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_usability.py",
            "--root",
            str(tmp_path),
            "--output-root",
            str(out_root),
            "--today",
            "2026-02-24",
            "--daily-tracker",
            "--live-ratings-csv",
            str(live_csv),
            "--alert-under",
            "0.7",
            "--target-rating",
            "0.8",
            "--fail-on-live-alert",
        ],
    )

    rc = dataset_usability.main()

    assert rc == 1
    assert (out_root / "reports" / "latest-dataset-usability-tracker.md").exists()
    assert (out_root / "reports" / "latest-dataset-usability-tracker.json").exists()
