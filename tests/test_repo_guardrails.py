from pathlib import Path
import json
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_kaggle_credentials_file_not_present():
    assert not (ROOT / "kaggle.json").exists(), (
        "kaggle.json must not exist in the repository root."
    )


def test_no_hardcoded_user_paths_in_scripts():
    offenders = []
    for path in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.sh")):
        if ".git" in path.parts or "tests" in path.parts or ".venv" in path.parts:
            continue
        content = path.read_text(encoding="utf-8")
        if "/Users/" in content:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        f"Hardcoded /Users paths found: {', '.join(sorted(offenders))}"
    )


def test_no_trust_remote_code_true_in_scripts():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts or "tests" in path.parts or ".venv" in path.parts:
            continue
        content = path.read_text(encoding="utf-8")
        if "trust_remote_code=True" in content:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        f"trust_remote_code=True found in: {', '.join(sorted(offenders))}"
    )


def test_manage_help_available():
    result = subprocess.run(
        ["bash", "manage.sh", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "scorecard" in result.stdout
    assert "weekly-plan" in result.stdout
    assert "pace" in result.stdout
    assert "sync" in result.stdout
    assert "sync-template" in result.stdout
    assert "doctor" in result.stdout
    assert "preflight" in result.stdout
    assert "quality" in result.stdout
    assert "dataset-usability" in result.stdout
    assert "usability-tracker" in result.stdout
    assert "campaign-pack" in result.stdout
    assert "campaign-run" in result.stdout
    assert "campaign-execute" in result.stdout
    assert "usability-benchmark" in result.stdout
    assert "publish-datasets" in result.stdout
    assert "smoke-live" in result.stdout
    assert "auth-doctor" in result.stdout
    assert "draft-ops" in result.stdout
    assert "draft-set" in result.stdout
    assert "--schedule-weeks" in result.stdout


def test_repo_root_has_no_top_level_python_scripts():
    root_scripts = sorted(path.name for path in ROOT.glob("*.py"))
    assert root_scripts == [], (
        "Move root Python files into kaggle_portfolio/ or a dedicated subdirectory. "
        f"Found: {', '.join(root_scripts)}"
    )


def test_medal_ops_health_workflow_exists_and_has_schedule():
    workflow = ROOT / ".github" / "workflows" / "medal-ops-health.yml"
    assert workflow.exists(), "Expected medal ops health workflow to exist."

    content = workflow.read_text(encoding="utf-8")
    assert "name: Medal Ops Health" in content
    assert "schedule:" in content
    assert 'cron: "10 9 * * 6"' in content
    assert "workflow_dispatch:" in content
    assert "mode:" in content
    assert "max_stale_days:" in content
    assert "min_quality_score:" in content
    assert "min_dataset_usability_score:" in content
    assert "live_alert_under:" in content
    assert "live_target_rating:" in content
    assert 'default: "0.8"' in content
    assert 'default: "1.0"' in content
    assert "max_overdue_scheduled:" in content
    assert "max_days_until_next_post:" in content
    assert 'default: "85"' in content
    # The medal_ops parser accepts shared flags before or after the subcommand,
    # so assert the doctor/sync steps exist with their key flags rather than
    # pinning an exact flag order that breaks on benign edits.
    assert "doctor" in content and "--strict" in content
    assert "--output-root /tmp/medal_ops_health" in content
    assert (
        "sync --dry-run" in content
        or "sync --output-root /tmp/medal_ops_health --dry-run" in content
    )
    assert "python -m kaggle_portfolio.quality.notebook_quality" in content
    assert "python -m kaggle_portfolio.datasets.dataset_usability" in content
    assert "dataset-usability.log" in content
    assert "dataset-usability-tracker.log" in content
    assert (
        "python -m kaggle_portfolio.ops.discussion_scheduler --health-check" in content
    )
    assert "draft-ops.log" in content
    assert "Open or update incident issue" in content


def test_ci_workflow_runs_preflight_gate_and_script_smokes():
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow.exists()
    content = workflow.read_text(encoding="utf-8")
    assert "bash manage.sh preflight" in content
    # PR CI must not enforce content-cadence SLAs (that is the scheduled
    # health workflow's job) and must not pin --today: a frozen date plus a
    # moving discussion queue once made every PR fail permanently.
    assert "--max-overdue-scheduled" in content
    preflight_step = content.split("bash manage.sh preflight")[1].split("- name:")[0]
    assert "--today" not in preflight_step
    assert "--no-pytest" in content
    assert "pytest -q --cov=." in content
    assert "python -m kaggle_portfolio.datasets.dataset_explore_generator" in content
    assert "python -m kaggle_portfolio.notebooks.competition_entry --help" in content


def test_live_smoke_workflow_exists_and_is_manual():
    workflow = ROOT / ".github" / "workflows" / "live-smoke.yml"
    assert workflow.exists(), "Expected live smoke workflow to exist."

    content = workflow.read_text(encoding="utf-8")
    assert "name: Live Smoke" in content
    assert "workflow_dispatch:" in content
    assert "schedule:" not in content
    assert "discussion_mode:" in content
    assert "include_live_datasets:" in content
    assert "kaggle-live-smoke" in content
    assert "bash manage.sh" in content
    assert "smoke-live" in content
    assert "--check-discussion-login" in content
    assert "--no-discussion" in content


def test_kaggle_session_cookie_is_gitignored():
    """The Playwright session-cookie file must never be committable."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "pi-automation/data/kaggle_storage_state.json"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, "kaggle_storage_state.json must be gitignored"


def test_medal_ops_history_is_tracked_not_ignored():
    """Daily snapshots must be committable so pace history accumulates."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "medal_ops/history/snapshot-sample.json"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    # returncode 1 == NOT ignored (what we want); 0 == ignored (fail)
    assert result.returncode == 1, "medal_ops/history/ must NOT be gitignored"


def test_telemetry_workflow_records_and_commits_snapshots():
    import yaml

    wf_path = ROOT / ".github" / "workflows" / "telemetry.yml"
    assert wf_path.exists(), "telemetry.yml must exist"
    wf = yaml.safe_load(wf_path.read_text(encoding="utf-8"))

    on = wf.get(True, wf.get("on"))  # PyYAML may parse bare `on:` as boolean True
    assert "schedule" in on
    assert "workflow_dispatch" in on
    assert wf.get("permissions", {}).get("contents") == "write"

    body = wf_path.read_text(encoding="utf-8")
    assert "medal_ops sync" in body
    assert (
        "medal_ops scorecard" in body
    )  # scorecard is what actually writes the snapshot
    assert "--dry-run" not in body
    assert "medal_ops digest" in body


def test_dataset_keywords_within_kaggle_limit():
    """Kaggle rejects dataset uploads with more than MAX_KEYWORDS keywords.

    Measured 2026-08-19 against the live API: 7 keywords fails with
    "You have exceeded the max category limit", 6 succeeds. Exceeding it means
    every push for that dataset silently stops working.
    """
    from kaggle_portfolio.manage_commands import MAX_KEYWORDS

    offenders = []
    for meta_path in sorted((ROOT / "datasets").glob("*/dataset-metadata.json")):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        keywords = payload.get("keywords") or []
        if len(keywords) > MAX_KEYWORDS:
            offenders.append(f"{meta_path.parent.name}: {len(keywords)}")

    assert not offenders, (
        f"Datasets exceed Kaggle's {MAX_KEYWORDS}-keyword limit: {offenders}"
    )


def test_deps_global_stays_inside_the_cli_edge():
    """No command module may reach manage_commands' process-wide Deps.

    ADR-0002 permits the edge to hold one constructed Deps; it does not permit
    command modules to default to it. That line was breached once already —
    notebook_quality imported is_skipped from manage_commands, and that helper
    called deps() — so the rule is enforced rather than trusted.
    """
    package = ROOT / "kaggle_portfolio"
    offenders = []
    for path in package.rglob("*.py"):
        if path.name == "manage_commands.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "manage_commands.deps(" in text or re.search(
            r"from kaggle_portfolio\.manage_commands import [^\n]*\bdeps\b", text
        ):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], (
        "these modules reach manage_commands' process-wide Deps: "
        f"{offenders}. Take deps as a parameter instead. Importing a function "
        "that ACCEPTS deps is fine; calling deps() from outside the edge is not."
    )


def test_benchmarks_registry_stays_callable():
    """BENCHMARKS[slug](data_dir, folds, write) is the documented contract.

    The package split briefly made the values dotted-path strings, so the
    subscript returned a str and calling it raised TypeError. Nothing caught it.
    """
    from kaggle_portfolio.notebooks.competition_lab import BENCHMARKS

    assert len(BENCHMARKS) >= 8
    for slug in BENCHMARKS:
        assert callable(BENCHMARKS[slug]), f"BENCHMARKS[{slug!r}] must be callable"


# Verified public on 2026-09-07 by fetching each notebook page anonymously:
#   curl -sL -o /dev/null -w '%{http_code}' \
#     https://www.kaggle.com/code/lorenzoscaturchio/<slug>
# 200 == live and visible logged out; 404 == private (or absent). The
# authenticated `kaggle kernels list --user` is NOT a privacy signal: it returns
# private kernels with full refs alongside public ones, which is what let
# is_private:true sit unnoticed on a medal-bearing notebook. Re-run the sweep
# before adding a slug here.
KNOWN_PUBLIC_KERNEL_SLUGS = frozenset(
    {
        "5-ways-your-cross-validation-lies-to-you",
        "adversarial-validation-trust-your-cv",
        "ai-data-jobs-market-explorer",
        "ai-data-jobs-skills-salaries-analysis",
        "ai-research-trends-explorer-v2",
        "competition-masterclass-full-ml-pipeline",
        "complete-guide-to-attention-mechanisms",
        "credit-card-fraud-detection-complete-ml-pipeline",
        "credit-card-fraud-eda-detection",
        "digit-recognizer-cnn-to-99-percent",
        "duckdb-on-kaggle-sql-analytics-without-a-database",
        "ecommerce-behavior-explorer-v2",
        "end-to-end-ml-pipeline-house-price-prediction",
        "ensemble-stacking-guide-win-kaggle-competitions",
        "feature-engineering-cookbook-50-techniques",
        "github-repo-metrics-explorer-v2",
        "gnn-guide-2026",
        "house-prices-complete-eda-feature-engineering",
        "image-segmentation-masterclass-u-net-to-segformer",
        "llm-fine-tuning-cookbook-lora-qlora",
        "med-gemma-challenge-medical-ai-eda-baseline",
        "mental-health-in-tech-policy-report-in-r-markdown",
        "mental-health-tech-explorer-v2",
        "ml-interview-qa-explorer-v2",
        "nlp-classification-pipeline-tf-idf-to-bert",
        "nlp-disaster-tweets-bert-guide",
        "optuna-tuning-a-practical-kaggle-guide",
        "playground-s6e6-stellar-classification",
        "polars-on-kaggle-the-complete-speed-guide",
        "programming-language-benchmarks-eda-v2",
        "rag-from-scratch",
        "shap-explainability-xai-masterclass",
        "spaceship-titanic-complete-ml-guide",
        "spotify-tracks-eda-popularity-prediction",
        "stock-market-analysis-prediction-with-python",
        "store-sales-time-series-forecasting-with-lightgbm",
        "student-performance-academic-eda",
        "student-performance-in-r-gpa-drivers",
        "tabular-eda-utilities-for-kaggle-projects",
        "time-series-forecasting-with-transformers",
        "titanic-ml-guide-zero-to-top-5-accuracy",
        "vesuvius-challenge-3d-surface-detection-eda",
    }
)


def _kernel_metadata_files():
    return sorted(ROOT.rglob("kernel-metadata.json"))


def test_public_notebooks_are_not_declared_private():
    """A public notebook must never carry is_private:true in this repo.

    `is_private` is pushed, not merely recorded: `manage.sh push` sends whatever
    the file says, so is_private:true on a live notebook unpublishes it. The
    notebook disappears from search, its votes stop counting toward medals, and
    nothing in the push output says so.

    store-sales-time-series-forecasting-with-lightgbm is the reason this test
    exists — it sat at is_private:true while live with 9 votes and a bronze
    medal, one `push` away from being silently delisted.
    """
    offenders = []
    for meta_path in _kernel_metadata_files():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        slug = str(payload.get("id", "")).split("/")[-1]
        if slug in KNOWN_PUBLIC_KERNEL_SLUGS and payload.get("is_private") is True:
            offenders.append(f"{meta_path.relative_to(ROOT)} ({slug})")

    assert not offenders, (
        "These notebooks are live and PUBLIC on Kaggle but declare "
        '"is_private": true. Pushing one unpublishes it — removing it from '
        "search and forfeiting its votes and any medal it earned. Set "
        '"is_private": false, or drop the slug from '
        f"KNOWN_PUBLIC_KERNEL_SLUGS if it was deliberately made private: {offenders}"
    )


def test_notebook_keywords_within_kaggle_limit():
    """Kaggle silently drops notebook keywords past MAX_KEYWORDS.

    Same 6-keyword cap the dataset test enforces, but notebooks fail quietly
    rather than loudly: `validate_dataset` rejects an over-cap dataset, while
    `validate_kernel` has no keyword check, so an over-cap notebook pushes
    "successfully" and simply loses every keyword after the sixth. The lost
    tags are the ones that would have made the notebook findable.
    """
    from kaggle_portfolio.manage_commands import MAX_KEYWORDS

    offenders = []
    for meta_path in _kernel_metadata_files():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        keywords = payload.get("keywords") or []
        if len(keywords) > MAX_KEYWORDS:
            offenders.append(f"{meta_path.relative_to(ROOT)}: {len(keywords)}")

    assert not offenders, (
        f"Notebooks exceed Kaggle's {MAX_KEYWORDS}-keyword cap; everything past "
        f"the {MAX_KEYWORDS}th is dropped on push without an error. Keep the "
        f"{MAX_KEYWORDS} most searchable terms: {offenders}"
    )


def test_hand_authored_explore_notebooks_are_protected_from_regeneration(repo_root):
    """A hand-authored explore notebook must survive `dataset_explore_generator --all`.

    This was a hardcoded name list that went stale: it protected spotify-tracks and
    mental-health-tech, while student-performance and ecommerce-behavior — both
    hand-authored, both carrying executed outputs — were left exposed to being
    overwritten by the generic template.
    """
    from kaggle_portfolio.datasets.dataset_explore_generator import is_hand_authored

    unprotected = []
    for ds_dir in sorted((repo_root / "datasets").iterdir()):
        if not ds_dir.is_dir():
            continue
        nb_path = ds_dir / "explore.ipynb"
        if not nb_path.exists():
            continue
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        # Saved outputs mean somebody executed it deliberately; the generator
        # never produces them.
        has_outputs = any(cell.get("outputs") for cell in nb.get("cells", []))
        if has_outputs and not is_hand_authored(ds_dir):
            unprotected.append(ds_dir.name)

    assert not unprotected, (
        "these explore notebooks carry executed outputs but would be overwritten by "
        f"`dataset_explore_generator --all`: {unprotected}. Add a build_notebook.py "
        'or set `"hand_authored": true` in the notebook metadata.'
    )


def test_kernel_ids_that_diverge_from_their_title_are_known(repo_root):
    """A push can move a kernel's slug to match its title. Keep that list explicit.

    Observed twice on 2026-09-08, both times landing on exactly slugify(title):

      student-performance-academic-eda
        -> student-performance-the-7-4-hour-sleep-optimum
      job-postings-nlp-salary-eda
        -> job-postings-nlp-salary-prediction-eda

    The old slugs 302-redirect and the votes carried, so this is not destructive —
    but every link shared elsewhere points at the old URL, and the repo's `id`
    silently goes stale, which is the drift #83 and #97 each had to repair.

    The mechanism is NOT established. `ecommerce-behavior` also diverges and did
    NOT move when pushed the same day, so "the slug always follows the title" is
    not a rule this repo has earned the right to assert. What is actionable is the
    exposure: these are the notebooks where a push MIGHT rename, and several carry
    votes. Adding one here is a decision to accept that risk, not a formality.
    """
    import re

    def slugify(title: str) -> str:
        return re.sub(
            r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        )

    known_divergent = {
        "ecommerce-behavior-explorer-v2",  # 2 votes; pushed, did not move
        "playground-s6e6-stellar-classification",  # 2 votes
        "complete-guide-to-attention-mechanisms",  # 1 vote
        "competition-masterclass-full-ml-pipeline",
        "end-to-end-ml-pipeline-house-price-prediction",
        "feature-engineering-cookbook-50-techniques",
        "llm-fine-tuning-cookbook-lora-qlora",  # 2 votes
        "rag-from-scratch",  # 5 votes -- the most exposed
        "time-series-forecasting-with-transformers",  # 2 votes
    }

    surprises = []
    for meta_path in sorted(repo_root.glob("**/kernel-metadata.json")):
        if ".venv" in str(meta_path):
            continue
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        slug = payload["id"].split("/")[-1]
        if slug != slugify(payload.get("title", "")) and slug not in known_divergent:
            surprises.append(f"{slug} (title: {payload.get('title')!r})")

    assert not surprises, (
        "these kernels' ids diverge from their slugified title and are not in the "
        f"known list, so a push may rename them and strand their URLs: {surprises}. "
        "Either align the title and id, or add the slug above with its vote count."
    )
