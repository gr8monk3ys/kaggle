import json
import sys
from datetime import date
import sys
from datetime import date
from pathlib import Path

import pytest

import dataset_metadata_sync as dms


def test_build_payload_defaults_and_citation():
    meta = {
        "id": "owner/sample-dataset",
        "title": "Sample Dataset",
        "authors": [{"name": "A Uthor", "bio": "Bio"}],
        "coverage": {
            "temporal_start_date": "2024-01-01",
            "temporal_end_date": "2025-01-01",
            "geospatial_coverage": "Global",
        },
        "doi": "Not assigned",
        "provenance": {
            "sources": ["Script A"],
            "collection_methodology": "Generated",
        },
    }

    payload = dms.build_payload(meta, "sample")

    assert payload.dataset_ref == "owner/sample-dataset"
    assert payload.author_name == "A Uthor"
    assert payload.author_bio == "Bio"
    assert payload.temporal_start_date == "2024-01-01"
    assert payload.temporal_end_date == "2025-01-01"
    assert payload.geospatial_coverage == "Global"
    assert payload.doi == ""
    assert payload.sources == ["Script A"]
    assert payload.collection_methodology == "Generated"
    assert payload.citations == [
        "Scaturchio, Lorenzo (2026). Sample Dataset. Kaggle Dataset. "
        "https://www.kaggle.com/datasets/owner/sample-dataset"
    ]


def test_build_payload_uses_force_doi():
    meta = {"id": "owner/ds", "title": "T"}
    payload = dms.build_payload(meta, "ds", force_doi="10.1234/example.doi")
    assert payload.doi == "10.1234/example.doi"


def test_discover_payloads_filters_dirs_and_refs(tmp_path: Path):
    datasets_root = tmp_path / "datasets"
    a = datasets_root / "a"
    b = datasets_root / "b"
    a.mkdir(parents=True)
    b.mkdir(parents=True)

    (a / "dataset-metadata.json").write_text(
        json.dumps({"id": "owner/a-ds", "title": "A"}),
        encoding="utf-8",
    )
    (b / "dataset-metadata.json").write_text(
        json.dumps({"id": "owner/b-ds", "title": "B"}),
        encoding="utf-8",
    )

    payloads_a = dms.discover_payloads(
        tmp_path,
        dataset_dirs={"a"},
        dataset_refs=None,
        force_doi=None,
    )
    assert [item.dataset_ref for item in payloads_a] == ["owner/a-ds"]

    payloads_b = dms.discover_payloads(
        tmp_path,
        dataset_dirs=None,
        dataset_refs={"owner/b-ds"},
        force_doi=None,
    )
    assert [item.dataset_ref for item in payloads_b] == ["owner/b-ds"]


def test_build_payload_requires_owner_slug():
    with pytest.raises(ValueError, match="owner/slug"):
        dms.build_payload({"id": "badref"}, "sample")
