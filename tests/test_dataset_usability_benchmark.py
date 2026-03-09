from __future__ import annotations

import json

import dataset_usability_benchmark as benchmark


def test_load_stringified_metadata_handles_string_encoded_json(tmp_path):
    payload = {
        "title": "Sample",
        "subtitle": "Quick subtitle",
        "description": "desc",
        "keywords": ["a", "b"],
        "licenses": [{"name": "CC0-1.0"}],
    }
    metadata_file = tmp_path / "dataset-metadata.json"
    metadata_file.write_text(json.dumps(json.dumps(payload)), encoding="utf-8")

    loaded = benchmark.load_stringified_metadata(metadata_file)

    assert loaded["title"] == "Sample"
    assert len(loaded["keywords"]) == 2


def test_parse_files_csv_extracts_starter_and_csv_flags():
    raw = (
        "name,size,creationDate\n"
        "README.md,100,2026-02-24\n"
        "data.csv,200,2026-02-24\n"
        "starter_analysis.ipynb,300,2026-02-24\n"
    )

    file_count, has_csv, has_starter = benchmark.parse_files_csv(raw)

    assert file_count == 3
    assert has_csv is True
    assert has_starter is True


def test_choose_exemplars_prefers_target_then_votes():
    rows = [
        benchmark.ListingRow(
            ref="u/a",
            title="A",
            size=1,
            last_updated="2026-02-24",
            download_count=10,
            vote_count=1,
            usability_rating=0.9,
        ),
        benchmark.ListingRow(
            ref="u/b",
            title="B",
            size=1,
            last_updated="2026-02-24",
            download_count=20,
            vote_count=5,
            usability_rating=1.0,
        ),
        benchmark.ListingRow(
            ref="u/c",
            title="C",
            size=1,
            last_updated="2026-02-24",
            download_count=30,
            vote_count=3,
            usability_rating=1.0,
        ),
    ]

    selected = benchmark.choose_exemplars(rows, target_rating=1.0, max_items=2)

    assert [item.ref for item in selected] == ["u/b", "u/c"]


def test_build_recommendations_flags_actionable_gaps():
    local_summary = {
        "public_pct": 0.0,
        "keyword_median": 2.0,
        "starter_asset_pct": 50.0,
        "license_pct": 80.0,
        "csv_pct": 100.0,
    }
    benchmark_summary = {
        "public_pct": 100.0,
        "keyword_median": 5.0,
        "starter_asset_pct": 100.0,
    }

    recs = benchmark.build_recommendations(local_summary, benchmark_summary)

    joined = "\n".join(recs).lower()
    assert "public variants" in joined
    assert "keyword coverage" in joined
    assert "starter assets" in joined
    assert "license metadata" in joined
