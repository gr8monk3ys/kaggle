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
    assert "quality" in result.stdout


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
    assert "doctor --strict" in content
    assert "notebook_quality.py" in content
    assert "sync --dry-run" in content
    assert "Open or update incident issue" in content


def test_ci_workflow_runs_notebook_quality_gate():
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow.exists()
    content = workflow.read_text(encoding="utf-8")
    assert "notebook_quality.py" in content
    assert "--fail-under-threshold" in content
