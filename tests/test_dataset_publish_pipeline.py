from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_portfolio.datasets import dataset_publish_pipeline as pipeline
from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.errors import CommandError
from kaggle_portfolio.shared.proc import summarize_output
from kaggle_portfolio.shared.kaggle_client import (
    Dataset,
    FakeKaggleClient,
    KaggleError,
    parse_csv,
)


def _candidate(
    rel: str,
    *,
    ref: str | None,
    score: int,
    live_state: str,
    eligible: bool,
) -> pipeline.PublishCandidate:
    return pipeline.PublishCandidate(
        rel_path=rel,
        dir_path=Path(rel),
        dataset_ref=ref,
        score=score,
        score_10=max(0, min(10, (score + 9) // 10)),
        tier="Good",
        live_state=live_state,
        eligible=eligible,
        blocked_reasons=[] if eligible else ["blocked"],
    )


def _client(csv_text: str = "", **kwargs) -> FakeKaggleClient:
    datasets = (
        [Dataset.from_row(row) for row in parse_csv(csv_text)] if csv_text else []
    )
    return FakeKaggleClient(datasets=datasets, **kwargs)


def test_fetch_live_refs_unions_private_and_public_listings():
    client = _client("ref,title\nowner/a,A\nowner/b,B\nowner/c,C\n")
    refs, err = pipeline.fetch_live_refs(client, "owner")
    assert err is None
    assert refs == {"owner/a", "owner/b", "owner/c"}


def test_fetch_live_refs_filters_non_owner_refs():
    client = _client("ref,title\nowner/a,A\nsomeone-else/z,Z\nowner/b,B\nanother/w,W\n")
    refs, err = pipeline.fetch_live_refs(client, "owner")
    assert err is None
    assert refs == {"owner/a", "owner/b"}


def test_fetch_live_refs_returns_empty_set_for_owner_with_no_datasets():
    refs, err = pipeline.fetch_live_refs(_client(), "owner")
    assert err is None
    assert refs == set()


def test_fetch_live_refs_reports_kaggle_failure():
    client = FakeKaggleClient(fail_with=KaggleError("403 Forbidden"))
    refs, err = pipeline.fetch_live_refs(client, "owner")
    assert refs is None
    assert "403 Forbidden" in err


def test_classify_live_state_handles_missing_lookup():
    assert pipeline.classify_live_state("owner/a", None) == "unknown"
    assert pipeline.classify_live_state(None, {"owner/a"}) == "unknown"
    assert pipeline.classify_live_state("owner/a", {"owner/a"}) == "live"
    assert pipeline.classify_live_state("owner/b", {"owner/a"}) == "draft"


def test_infer_owner_uses_majority_dataset_ref_owner():
    candidates = [
        _candidate(
            "datasets/a", ref="owner-one/a", score=90, live_state="draft", eligible=True
        ),
        _candidate(
            "datasets/b", ref="owner-one/b", score=90, live_state="draft", eligible=True
        ),
        _candidate(
            "datasets/c", ref="owner-two/c", score=90, live_state="draft", eligible=True
        ),
    ]

    assert pipeline.infer_owner(candidates) == "owner-one"


def test_select_targets_respects_draft_mode_and_max_items():
    candidates = [
        _candidate(
            "datasets/a", ref="owner/a", score=90, live_state="draft", eligible=True
        ),
        _candidate(
            "datasets/b", ref="owner/b", score=90, live_state="live", eligible=True
        ),
        _candidate(
            "datasets/c", ref="owner/c", score=90, live_state="draft", eligible=False
        ),
    ]

    draft_only = pipeline.select_targets(candidates, draft_only=True, max_items=0)
    assert [item.rel_path for item in draft_only] == ["datasets/a"]

    all_mode = pipeline.select_targets(candidates, draft_only=False, max_items=1)
    assert [item.rel_path for item in all_mode] == ["datasets/a"]


def test_summarize_output_prefers_real_error_line():
    message = summarize_output(
        "warning one\nwarning two\n",
        "401 Client Error: Unauthorized for url: https://www.kaggle.com/api/v1/blobs/upload\n"
        "  warnings.warn(\n",
    )
    assert message.startswith("401 Client Error: Unauthorized")


# -- main(): the `publish-datasets` command end to end ------------------------
#
# Everything above exercises the helpers; these drive the command the README
# promises ("publishes datasets that pass the usability gate, dry run without
# --apply") against a temporary repo and the in-memory Kaggle.


def _write_bundle(
    root: Path,
    name: str,
    *,
    ref: str,
    strong: bool = True,
    resources: list[str] | None = None,
) -> Path:
    """A dataset folder that scores >= 85 when ``strong``, < 55 otherwise."""
    ds_dir = root / "datasets" / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict = {"title": name.title(), "id": ref}
    if strong:
        metadata.update(
            {
                "subtitle": "Clean sample",
                "description": "Detailed description. " * 60,
                "licenses": [{"name": "CC0-1.0"}],
                "keywords": ["ml", "nlp", "classification", "baseline", "analysis"],
            }
        )
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
                f"**Kaggle:** [{ref}](https://www.kaggle.com/datasets/{ref})",
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
        (ds_dir / "README.md").write_text(readme, encoding="utf-8")
        (ds_dir / "create_dataset.py").write_text("print('ok')\n", encoding="utf-8")
        (ds_dir / "explore.ipynb").write_text("{}", encoding="utf-8")
        (ds_dir / "kernel-metadata.json").write_text(
            json.dumps(
                {"id": "u/explore", "title": "Explore", "code_file": "explore.ipynb"}
            ),
            encoding="utf-8",
        )
        for filename in ("sample.csv", "sample.parquet"):
            (ds_dir / filename).write_text("a,b\n1,2\n", encoding="utf-8")
    else:
        metadata.update({"description": "short", "licenses": [], "keywords": []})
    if resources is not None:
        metadata["resources"] = [{"path": rel} for rel in resources]
    (ds_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return ds_dir


def _repo(tmp_path: Path) -> Path:
    """owner/fresh: eligible draft. owner/old: eligible but live. owner/weak: below gate."""
    _write_bundle(tmp_path, "fresh", ref="owner/fresh")
    _write_bundle(tmp_path, "old", ref="owner/old")
    _write_bundle(tmp_path, "weak", ref="owner/weak", strong=False)
    return tmp_path


def _run(root: Path, client: FakeKaggleClient, *argv: str) -> int:
    deps = Deps.for_test(root, client=client)
    return pipeline.main(["--root", str(root), *argv], deps=deps)


def _published(client: FakeKaggleClient) -> list[str]:
    return [
        Path(args[0]).name for name, args in client.calls if name == "publish_dataset"
    ]


def test_main_dry_run_selects_eligible_drafts_and_publishes_nothing(tmp_path):
    root = _repo(tmp_path)
    client = _client("ref,title\nowner/old,Old\n")
    report = tmp_path / "out" / "report.json"

    code = _run(root, client, "--report-json", str(report))

    assert code == 0
    assert client.calls == []
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["owner"] == "owner"  # inferred from the dataset refs
    assert payload["apply"] is False
    assert payload["mode"] == "draft-only"
    assert payload["selected_count"] == 1
    states = {
        c["rel_path"]: (c["live_state"], c["eligible"]) for c in payload["candidates"]
    }
    assert states == {
        "datasets/fresh": ("draft", True),
        "datasets/old": ("live", True),
        "datasets/weak": ("draft", False),
    }
    assert payload["results"] == []


def test_main_apply_publishes_only_eligible_drafts(tmp_path):
    root = _repo(tmp_path)
    client = _client("ref,title\nowner/old,Old\n")
    report = tmp_path / "report.json"

    code = _run(root, client, "--apply", "--report-json", str(report))

    assert code == 0
    assert _published(client) == ["fresh"]
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["publish"] == {"success": 1, "failed": 0}
    assert [(r["rel_path"], r["ok"]) for r in payload["results"]] == [
        ("datasets/fresh", True)
    ]


def test_main_apply_all_includes_live_datasets_and_honours_max_items(tmp_path):
    root = _repo(tmp_path)
    client = _client("ref,title\nowner/old,Old\n")

    assert _run(root, client, "--apply", "--all") == 0
    assert _published(client) == ["fresh", "old"]

    capped = _client("ref,title\nowner/old,Old\n")
    assert _run(root, capped, "--apply", "--all", "--max-items", "1") == 0
    assert _published(capped) == ["fresh"]


def test_main_min_score_is_the_gate_that_holds_back_a_weak_dataset(tmp_path):
    root = _repo(tmp_path)
    report = tmp_path / "report.json"

    default = _client("ref,title\nowner/old,Old\n")
    assert _run(root, default, "--apply", "--report-json", str(report)) == 0
    assert "weak" not in _published(default)
    weak = next(
        c
        for c in json.loads(report.read_text(encoding="utf-8"))["candidates"]
        if c["rel_path"] == "datasets/weak"
    )
    assert any("< min-score 85" in reason for reason in weak["blocked_reasons"])

    lowered = _client("ref,title\nowner/old,Old\n")
    assert _run(root, lowered, "--apply", "--min-score", "0") == 0
    assert _published(lowered) == ["fresh", "weak"]


def test_main_draft_only_refuses_without_live_listing(tmp_path, capsys):
    root = _repo(tmp_path)
    client = FakeKaggleClient(fail_with=KaggleError("401 Unauthorized"))

    code = _run(root, client, "--apply")

    assert code == 1
    assert _published(client) == []
    out = capsys.readouterr().out
    assert "401 Unauthorized" in out
    assert "Cannot run in draft-only mode" in out


def test_main_apply_reports_failure_when_declared_files_are_missing(tmp_path):
    _write_bundle(tmp_path, "unbuilt", ref="owner/unbuilt", resources=["data.csv"])
    client = _client()
    report = tmp_path / "report.json"

    code = _run(tmp_path, client, "--apply", "--report-json", str(report))

    assert code == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["publish"] == {"success": 0, "failed": 1}
    [result] = payload["results"]
    assert result["ok"] is False
    assert "missing data.csv" in result["detail"]


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--min-score", "101"], "--min-score"),
        (["--min-score", "-1"], "--min-score"),
        (["--max-items", "-1"], "--max-items"),
    ],
)
def test_main_rejects_out_of_range_arguments(tmp_path, argv, message):
    _repo(tmp_path)
    with pytest.raises(CommandError, match=message):
        _run(tmp_path, _client(), *argv)


def test_main_errors_when_there_are_no_datasets(tmp_path):
    with pytest.raises(CommandError, match="No dataset directories"):
        _run(tmp_path, _client())
