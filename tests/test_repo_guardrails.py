from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_kaggle_credentials_file_not_present():
    assert not (ROOT / "kaggle.json").exists(), "kaggle.json must not exist in the repository root."


def test_no_hardcoded_user_paths_in_scripts():
    offenders = []
    for path in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.sh")):
        if ".git" in path.parts or "tests" in path.parts or ".venv" in path.parts:
            continue
        content = path.read_text(encoding="utf-8")
        if "/Users/" in content:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"Hardcoded /Users paths found: {', '.join(sorted(offenders))}"


def test_no_trust_remote_code_true_in_scripts():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts or "tests" in path.parts or ".venv" in path.parts:
            continue
        content = path.read_text(encoding="utf-8")
        if "trust_remote_code=True" in content:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"trust_remote_code=True found in: {', '.join(sorted(offenders))}"


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
    assert 'name: Medal Ops Health' in content
    assert "schedule:" in content
    assert 'cron: "10 9 * * *"' in content
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
    assert "doctor --output-root /tmp/medal_ops_health --strict" in content
    assert "python -m kaggle_portfolio.quality.notebook_quality" in content
    assert "python -m kaggle_portfolio.datasets.dataset_usability" in content
    assert "dataset-usability.log" in content
    assert "dataset-usability-tracker.log" in content
    assert "python -m kaggle_portfolio.ops.discussion_scheduler --health-check" in content
    assert "draft-ops.log" in content
    assert "sync --output-root /tmp/medal_ops_health --dry-run" in content
    assert "Open or update incident issue" in content


def test_ci_workflow_runs_preflight_gate_and_script_smokes():
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow.exists()
    content = workflow.read_text(encoding="utf-8")
    assert "bash manage.sh preflight" in content
    assert "--strict-doctor" in content
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
