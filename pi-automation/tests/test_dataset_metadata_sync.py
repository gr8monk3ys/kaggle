import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, "pi-automation/scripts")

import dataset_metadata_sync as dms


class _FakeLocator:
    def __init__(self, count: int = 0):
        self._count = count

    @property
    def first(self):
        return self

    def count(self) -> int:
        return self._count


class _FakePage:
    def __init__(self, *, url: str, signed_out: bool):
        self.url = url
        self.signed_out = signed_out

    def get_by_role(self, role: str, name=None):
        pattern = getattr(name, "pattern", "") if name is not None else ""
        if not self.signed_out:
            return _FakeLocator(0)
        if role in {"link", "button"} and "sign in" in str(pattern).lower():
            return _FakeLocator(1)
        if role == "link" and "register" in str(pattern).lower():
            return _FakeLocator(1)
        return _FakeLocator(0)

    def locator(self, selector: str):
        if self.signed_out and "/account/login" in selector:
            return _FakeLocator(1)
        return _FakeLocator(0)


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
        f"Scaturchio, Lorenzo ({date.today().year}). Sample Dataset. Kaggle Dataset. "
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


def test_is_authenticated_false_for_login_url():
    page = _FakePage(url="https://www.kaggle.com/account/login", signed_out=False)
    assert dms.is_authenticated(page) is False


def test_is_authenticated_false_when_sign_in_prompt_visible():
    page = _FakePage(url="https://www.kaggle.com/datasets/owner/ds", signed_out=True)
    assert dms.is_authenticated(page) is False


def test_is_authenticated_true_for_signed_in_page():
    page = _FakePage(url="https://www.kaggle.com/datasets/owner/ds/settings", signed_out=False)
    assert dms.is_authenticated(page) is True


def test_storage_state_has_kaggle_cookie_true(tmp_path: Path):
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "cookies": [
                    {"name": "foo", "domain": ".example.com"},
                    {"name": "bar", "domain": ".kaggle.com"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert dms.storage_state_has_kaggle_cookie(state) is True


def test_storage_state_has_kaggle_cookie_false(tmp_path: Path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"cookies": [{"name": "foo", "domain": ".example.com"}]}), encoding="utf-8")
    assert dms.storage_state_has_kaggle_cookie(state) is False
