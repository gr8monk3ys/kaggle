from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from kaggle_portfolio.shared.kaggle_utils import (
    has_kaggle_cli as shared_has_kaggle_cli,
    kaggle_command,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("KAGGLE_DIR", str(PACKAGE_ROOT))).resolve()
PI_SCRIPTS = PACKAGE_ROOT / "pi-automation" / "scripts"
DEFAULT_CREDENTIALS = Path.home() / ".kaggle" / "kaggle.json"
LOCAL_CREDENTIALS = ROOT / "kaggle.json"
METADATA_NAMES = {"kernel-metadata.json", "dataset-metadata.json"}
SKIP_DIRS = {
    ".claude",
    ".git",
    ".venv",
    ".pytest_cache",
    ".playwright-cli",
    ".playwright-mcp",
    "__pycache__",
}
SUSPICIOUS_PATTERN = re.compile(
    r"(password|secret|api_key|kgat_|kaggle_token)", re.IGNORECASE
)


def is_skipped(path: Path, root: Path = ROOT) -> bool:
    """True when path lies inside a skipped directory, judged relative to root.

    Relative, not absolute: the checkout itself can sit under a hidden
    directory (agent worktrees live in .claude/worktrees/), and matching on
    absolute parts would then skip every file in the repo.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    return any(part in SKIP_DIRS for part in rel.parts)


TRUTHY = {"1", "true", "yes", "on"}

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
RESET = "\033[0m"


def discover_notebook_dirs() -> list[str]:
    items: list[str] = []
    for meta in sorted(ROOT.rglob("kernel-metadata.json")):
        if is_skipped(meta):
            continue
        try:
            rel = meta.parent.relative_to(ROOT)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "datasets":
            continue
        items.append(str(rel))
    return items


def discover_dataset_dirs() -> list[str]:
    items: list[str] = []
    for meta in sorted(ROOT.rglob("dataset-metadata.json")):
        if is_skipped(meta):
            continue
        try:
            items.append(str(meta.parent.relative_to(ROOT)))
        except ValueError:
            continue
    return items


NOTEBOOK_DIRS = discover_notebook_dirs()
DATASET_DIRS = discover_dataset_dirs()


def print_usage() -> None:
    print("Usage: ./manage.sh <command> [options]")
    print("")
    print("Commands:")
    for command in COMMANDS:
        if command.hidden:
            continue
        label = command.name if not command.args else f"{command.name} {command.args}"
        print(f"  {label:<34} {command.description}")


def has_kaggle_cli() -> bool:
    return shared_has_kaggle_cli()


def require_kaggle_cli() -> None:
    if not has_kaggle_cli():
        raise SystemExit("Error: kaggle CLI not found. Install it with: pip install kaggle")


def has_kaggle_credentials() -> tuple[bool, list[str]]:
    sources: list[str] = []

    env_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    env_user = os.environ.get("KAGGLE_USERNAME", "").strip()
    env_key = os.environ.get("KAGGLE_KEY", "").strip()

    if env_token:
        sources.append("environment-token")
    if env_user and env_key:
        sources.append("environment")
    if DEFAULT_CREDENTIALS.exists():
        sources.append(str(DEFAULT_CREDENTIALS))
    if LOCAL_CREDENTIALS.exists():
        sources.append(str(LOCAL_CREDENTIALS))

    return bool(sources), sources


def require_kaggle_credentials() -> None:
    ok, _sources = has_kaggle_credentials()
    if ok:
        return
    raise SystemExit(
        "Error: Kaggle credentials not found.\n"
        f"Create {DEFAULT_CREDENTIALS} (recommended) and run:\n"
        f"  chmod 600 {DEFAULT_CREDENTIALS}"
    )


def ensure_kaggle_ready() -> None:
    require_kaggle_cli()
    require_kaggle_credentials()


def kaggle_cmd(*args: str, check: bool = False, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [*kaggle_command(), *args]
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=capture_output,
        check=check,
    )


def run_script(path: Path, args: list[str]) -> int:
    result = subprocess.run([sys.executable, str(path), *args], cwd=PACKAGE_ROOT, check=False)
    return result.returncode


def run_module(module: str, args: list[str]) -> int:
    result = subprocess.run([sys.executable, "-m", module, *args], cwd=PACKAGE_ROOT, check=False)
    return result.returncode


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def git_head_available() -> bool:
    repo_probe = git_run("rev-parse", "--is-inside-work-tree")
    if repo_probe.returncode != 0:
        return False
    return git_run("rev-parse", "--verify", "HEAD").returncode == 0


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUTHY


def head_payload(path: Path) -> dict | None:
    if not env_truthy("VALIDATE_ENFORCE_ID_BASELINE") or env_truthy("MANAGE_ALLOW_ID_CHANGE"):
        return None
    if not git_head_available():
        return None
    rel = rel_path(path).replace(os.sep, "/")
    tracked = git_run("ls-files", "--error-unmatch", rel)
    if tracked.returncode != 0:
        return None
    result = git_run("show", f"HEAD:{rel}")
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def resolve_target(target: str) -> Path:
    path = Path(target)
    if path.is_absolute():
        return path.resolve()

    direct = (ROOT / target).resolve()
    if direct.exists():
        return direct

    matches: list[Path] = []
    for rel in NOTEBOOK_DIRS + DATASET_DIRS:
        rel_path = Path(rel)
        if str(rel_path) == target or rel_path.name == target:
            matches.append((ROOT / rel_path).resolve())

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        joined = ", ".join(str(match.relative_to(ROOT)) for match in matches[:10])
        raise SystemExit(f"Ambiguous target '{target}'. Matches: {joined}")
    return direct


def in_scope(path: Path, scope: Path | None) -> bool:
    resolved = path.resolve()
    if scope is None:
        return True
    if scope.is_file():
        return resolved == scope
    try:
        resolved.relative_to(scope)
    except ValueError:
        return False
    return True


def iter_metadata_files(scope: Path | None) -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*-metadata.json"):
        if path.name not in METADATA_NAMES:
            continue
        if is_skipped(path):
            continue
        if in_scope(path, scope):
            files.append(path)
    return sorted(files)


def validate_kernel(path: Path, payload: dict, raw_text: str) -> list[str]:
    errors: list[str] = []
    required = ("id", "title", "code_file", "language", "kernel_type", "is_private")
    for field in required:
        if field not in payload:
            errors.append(f"missing '{field}'")

    ident = str(payload.get("id", "")).strip()
    title = str(payload.get("title", "")).strip()
    code_file = str(payload.get("code_file", "")).strip()

    if "id" in payload and not ident:
        errors.append("missing 'id'")
    elif ident and ("/" not in ident or " " in ident):
        errors.append(f"id '{ident}' must use owner/slug format with no spaces")
    else:
        baseline = head_payload(path)
        baseline_id = str(baseline.get("id", "")).strip() if baseline else ""
        if baseline_id and ident and baseline_id != ident:
            errors.append(
                f"id changed from '{baseline_id}' to '{ident}' relative to git HEAD; "
                "pushing can create a duplicate Kaggle notebook "
                "(set MANAGE_ALLOW_ID_CHANGE=1 to override)"
            )

    if "title" in payload and not title:
        errors.append("missing 'title'")
    elif title and not 6 <= len(title) <= 70:
        errors.append(f"title length {len(title)} (must be 6-70)")

    if "code_file" in payload and not code_file:
        errors.append("missing 'code_file'")
    elif code_file and not (path.parent / code_file).exists():
        errors.append(f"code_file '{code_file}' not found in {path.parent.name}/")

    for field in ("dataset_sources", "kernel_sources"):
        value = payload.get(field)
        if value is not None and not isinstance(value, list):
            errors.append(f"'{field}' must be a list")

    if SUSPICIOUS_PATTERN.search(raw_text):
        errors.append("possible credential in metadata — review before pushing")
    return errors


def validate_dataset(path: Path, payload: dict, raw_text: str) -> list[str]:
    errors: list[str] = []
    required = ("id", "title", "licenses", "resources", "authors", "coverage", "provenance")
    for field in required:
        if field not in payload:
            errors.append(f"missing '{field}'")

    ident = str(payload.get("id", "")).strip()
    title = str(payload.get("title", "")).strip()
    licenses = payload.get("licenses")
    resources = payload.get("resources")
    authors = payload.get("authors")
    coverage = payload.get("coverage")
    provenance = payload.get("provenance")

    if "id" in payload and not ident:
        errors.append("missing 'id'")
    elif ident and ("/" not in ident or " " in ident):
        errors.append(f"id '{ident}' must use owner/slug format with no spaces")

    if "title" in payload and not title:
        errors.append("missing 'title'")

    if not isinstance(licenses, list) or not licenses:
        errors.append("missing non-empty 'licenses' list")
    else:
        for index, item in enumerate(licenses, start=1):
            if not isinstance(item, dict) or not str(item.get("name", "")).strip():
                errors.append(f"license #{index} missing 'name'")

    if not isinstance(resources, list) or not resources:
        errors.append("missing non-empty 'resources' list")
    else:
        for index, item in enumerate(resources, start=1):
            if not isinstance(item, dict):
                errors.append(f"resource #{index} must be an object")
                continue
            resource_path = str(item.get("path", "")).strip()
            if not resource_path:
                errors.append(f"resource #{index} missing 'path'")
                continue
            if not (path.parent / resource_path).exists():
                errors.append(
                    f"resource path '{resource_path}' not found in {path.parent.name}/"
                )
            if not str(item.get("description", "")).strip():
                errors.append(f"resource #{index} missing 'description'")

            schema = item.get("schema")
            if not isinstance(schema, dict):
                errors.append(f"resource #{index} missing 'schema'")
                continue

            fields = schema.get("fields")
            if not isinstance(fields, list) or not fields:
                errors.append(f"resource #{index} missing non-empty 'schema.fields'")
                continue

            for field_index, field_payload in enumerate(fields, start=1):
                if not isinstance(field_payload, dict):
                    errors.append(
                        f"resource #{index} field #{field_index} must be an object"
                    )
                    continue
                for field_name in ("name", "title", "description", "type"):
                    if not str(field_payload.get(field_name, "")).strip():
                        errors.append(
                            f"resource #{index} field #{field_index} missing '{field_name}'"
                        )

    if not isinstance(authors, list) or not authors:
        errors.append("missing non-empty 'authors' list")
    else:
        for index, item in enumerate(authors, start=1):
            if not isinstance(item, dict) or not str(item.get("name", "")).strip():
                errors.append(f"author #{index} missing 'name'")

    if not isinstance(coverage, dict):
        errors.append("missing 'coverage' object")
    else:
        for field in ("temporal_start_date", "temporal_end_date", "geospatial_coverage"):
            if not str(coverage.get(field, "")).strip():
                errors.append(f"coverage missing '{field}'")

    if not isinstance(provenance, dict):
        errors.append("missing 'provenance' object")
    else:
        sources = provenance.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append("provenance missing non-empty 'sources' list")
        else:
            for index, item in enumerate(sources, start=1):
                if not str(item).strip():
                    errors.append(f"provenance source #{index} is empty")
        if not str(provenance.get("collection_methodology", "")).strip():
            errors.append("provenance missing 'collection_methodology'")

    if SUSPICIOUS_PATTERN.search(raw_text):
        errors.append("possible credential in metadata — review before pushing")
    return errors


def cmd_validate(args: list[str]) -> int:
    scope = None
    if args:
        scope = resolve_target(args[0])
        if not scope.exists():
            print(f"{RED}Error:{RESET} validation target not found: {args[0]}", file=sys.stderr)
            return 1

    print(f"{BLUE}=== Validating metadata files ==={RESET}")
    print("")

    checked = 0
    errors = 0
    files = iter_metadata_files(scope)
    if not files:
        print("No metadata files found to validate.")
        return 1

    for meta in files:
        checked += 1
        rel = rel_path(meta)
        try:
            raw = meta.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"  FAIL {rel}")
            print(f"       -> unreadable file: {exc}")
            errors += 1
            continue

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  FAIL {rel} — invalid JSON")
            errors += 1
            continue

        if not isinstance(payload, dict):
            print(f"  FAIL {rel}")
            print("       -> metadata must be a JSON object")
            errors += 1
            continue

        file_errors = (
            validate_kernel(meta, payload, raw)
            if meta.name == "kernel-metadata.json"
            else validate_dataset(meta, payload, raw)
        )
        if not file_errors:
            print(f"  OK   {rel}")
            continue

        print(f"  FAIL {rel}")
        for err in file_errors:
            print(f"       -> {err}")
        errors += 1

    print("")
    print(f"Checked {checked} files — {errors} error(s)")
    if errors:
        print("Validation FAILED. Fix errors before pushing.")
        return 1
    print("All metadata files valid.")
    return 0


def validate_for_push(args: list[str], *, enforce_id_baseline: bool) -> int:
    previous = os.environ.get("VALIDATE_ENFORCE_ID_BASELINE")
    if enforce_id_baseline:
        os.environ["VALIDATE_ENFORCE_ID_BASELINE"] = "1"
    try:
        return cmd_validate(args)
    finally:
        if enforce_id_baseline:
            if previous is None:
                os.environ.pop("VALIDATE_ENFORCE_ID_BASELINE", None)
            else:
                os.environ["VALIDATE_ENFORCE_ID_BASELINE"] = previous


def cmd_status(_: list[str]) -> int:
    print(f"{BLUE}=== Kaggle Portfolio Status ==={RESET}")
    print("")
    print(f"{YELLOW}Notebooks:{RESET}")
    result = kaggle_cmd("kernels", "list", "--mine", "--page-size", "50", check=False, capture_output=True)
    if result.stdout:
        print("\n".join(result.stdout.splitlines()[:30]))
    print("")
    print(f"{YELLOW}Datasets:{RESET}")
    result = kaggle_cmd("datasets", "list", "-m", check=False, capture_output=True)
    if result.stdout:
        print("\n".join(result.stdout.splitlines()[:20]))
    print("")
    print(f"{YELLOW}Local directories:{RESET}")
    print(f"  Notebooks: {len(NOTEBOOK_DIRS)}")
    print(f"  Datasets:  {len(DATASET_DIRS)}")
    return 0


def push_dataset(path: Path) -> int:
    create = ["datasets", "create", "-p", str(path), "--dir-mode", "zip"]
    version = ["datasets", "version", "-p", str(path), "-m", "Updated content", "--dir-mode", "zip"]
    result = kaggle_cmd(*version, check=False)
    if result.returncode == 0:
        return 0
    return kaggle_cmd(*create, check=False).returncode


def cmd_push(args: list[str]) -> int:
    if not args:
        raise SystemExit("Usage: ./manage.sh push <directory>")
    target = args[0]
    path = resolve_target(target)
    if not path.exists():
        print(f"{RED}Error:{RESET} path not found: {target}", file=sys.stderr)
        return 1
    print(f"{YELLOW}Running validation for {target}...{RESET}")
    enforce_id_baseline = (path / "kernel-metadata.json").exists()
    if validate_for_push([str(path)], enforce_id_baseline=enforce_id_baseline) != 0:
        print(f"{RED}Fix validation errors before pushing.{RESET}")
        return 1
    if (path / "dataset-metadata.json").exists():
        print(f"Pushing dataset: {target}")
        return push_dataset(path)
    if (path / "kernel-metadata.json").exists():
        print(f"Pushing notebook: {target}")
        return kaggle_cmd("kernels", "push", "-p", str(path), check=False).returncode
    raise SystemExit(f"Error: No metadata found in {path}")


def cmd_push_nb(_: list[str]) -> int:
    print(f"{BLUE}=== Pushing All Notebooks ==={RESET}")
    print(f"{YELLOW}Running pre-push validation...{RESET}")
    if validate_for_push([], enforce_id_baseline=True) != 0:
        print(f"{RED}Fix validation errors before pushing.{RESET}")
        return 1
    print("")
    success = 0
    failed = 0
    for rel in NOTEBOOK_DIRS:
        path = ROOT / rel
        if not (path / "kernel-metadata.json").exists():
            print(f"  {YELLOW}SKIP{RESET} {rel} (no kernel-metadata.json)")
            continue
        print(f"  Pushing {rel}... ", end="", flush=True)
        result = kaggle_cmd("kernels", "push", "-p", str(path), check=False, capture_output=True)
        if result.returncode == 0:
            print(f"{GREEN}OK{RESET}")
            success += 1
        else:
            print(f"{RED}FAILED{RESET}: {result.stderr.strip() or result.stdout.strip()}")
            failed += 1
    print("")
    print(f"Results: {GREEN}{success} succeeded{RESET}, {RED}{failed} failed{RESET}")
    return 0 if failed == 0 else 1


def cmd_push_ds(_: list[str]) -> int:
    print(f"{BLUE}=== Pushing All Datasets ==={RESET}")
    print(f"{YELLOW}Running pre-push validation...{RESET}")
    if cmd_validate([]) != 0:
        print(f"{RED}Fix validation errors before pushing.{RESET}")
        return 1
    print("")
    success = 0
    failed = 0
    for rel in DATASET_DIRS:
        path = ROOT / rel
        if not (path / "dataset-metadata.json").exists():
            print(f"  {YELLOW}SKIP{RESET} {rel} (no dataset-metadata.json)")
            continue
        print(f"  Pushing {rel}... ", end="", flush=True)
        result = kaggle_cmd(
            "datasets",
            "version",
            "-p",
            str(path),
            "-m",
            "Updated content",
            "--dir-mode",
            "zip",
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"{GREEN}UPDATED{RESET}")
            success += 1
            continue
        create = kaggle_cmd(
            "datasets",
            "create",
            "-p",
            str(path),
            "--dir-mode",
            "zip",
            check=False,
            capture_output=True,
        )
        if create.returncode == 0:
            print(f"{GREEN}CREATED{RESET}")
            success += 1
        else:
            print(f"{RED}FAILED{RESET}: {create.stderr.strip() or create.stdout.strip()}")
            failed += 1
    print("")
    print(f"Results: {GREEN}{success} succeeded{RESET}, {RED}{failed} failed{RESET}")
    return 0 if failed == 0 else 1


def cmd_push_all(args: list[str]) -> int:
    rc = cmd_push_nb(args)
    if rc != 0:
        return rc
    return cmd_push_ds(args)


def cmd_votes(_: list[str]) -> int:
    bronze_t = 5
    silver_t = 20
    gold_t = 50
    print(f"{BLUE}=== Vote Counts & Medal Threshold Dashboard ==={RESET}")
    print("")
    print(
        f"{YELLOW}Medal thresholds:{RESET}  Bronze >={bronze_t}  Silver >={silver_t}  Gold >={gold_t}"
    )
    print("")
    raw = kaggle_cmd(
        "kernels",
        "list",
        "--mine",
        "--page-size",
        "100",
        "--kernel-type",
        "notebook",
        "--csv",
        check=False,
        capture_output=True,
    )
    if raw.returncode != 0:
        print(f"{RED}Failed to fetch kernel list from Kaggle.{RESET}")
        return 1

    csv_lines = "\n".join(
        line
        for line in raw.stdout.splitlines()
        if not line.startswith("Warning:") and not line.startswith("/")
    )
    rows = list(csv.DictReader(io.StringIO(csv_lines)))
    if not rows:
        print("No kernels found.")
        return 0

    sample = rows[0] if rows else {}
    vote_col = next(
        (k for k in sample if k and ("vote" in k.lower() or "upvote" in k.lower())),
        None,
    )
    ref_col = next((k for k in sample if k and "ref" in k.lower()), None)
    title_col = next((k for k in sample if k and "title" in k.lower()), None)

    def is_public_notebook(row: dict[str, str]) -> bool:
        ref = str(row.get(ref_col, "")).strip() if ref_col else ""
        title = str(row.get(title_col, "")).strip().lower() if title_col else ""
        return bool(ref) and title != "[private notebook]"

    rows = [row for row in rows if is_public_notebook(row)]
    if not rows:
        print("No public notebooks found.")
        return 0

    def next_tier(votes: int) -> tuple[str, int]:
        if votes < bronze_t:
            return ("Bronze", bronze_t)
        if votes < silver_t:
            return ("Silver", silver_t)
        if votes < gold_t:
            return ("Gold", gold_t)
        return ("GOLD+", gold_t)

    def medal_color(votes: int) -> str:
        if votes >= gold_t:
            return CYAN
        if votes >= silver_t:
            return GREEN
        if votes >= bronze_t:
            return YELLOW
        return RESET

    print(f"{'Notebook':<45} {'Votes':>6}  {'Tier':<8} {'Next':<8} {'Gap':>4}")
    print("-" * 78)

    def sort_key(row: dict[str, str]) -> tuple[int, int]:
        votes = int(row.get(vote_col or "", 0) or 0)
        _, threshold = next_tier(votes)
        return (-votes, threshold - votes)

    for row in sorted(rows, key=sort_key):
        ref = (
            row.get(ref_col, row.get(title_col, row.get("ref", "unknown")))
            if (ref_col or title_col)
            else row.get("ref", "unknown")
        )
        votes = int(row.get(vote_col or "", 0) or 0) if vote_col else 0
        tier_name, tier_thresh = next_tier(votes)
        gap = tier_thresh - votes
        if votes >= gold_t:
            current = "GOLD"
        elif votes >= silver_t:
            current = "Silver"
        elif votes >= bronze_t:
            current = "Bronze"
        else:
            current = "—"
        gap_color = GREEN if 0 < gap <= 3 else YELLOW if 0 < gap <= 10 else RESET
        name = str(ref).split("/")[-1][:44]
        print(
            f"{medal_color(votes)}{name:<45}{RESET} {votes:>6}  {current:<8} "
            f"{tier_name:<8} {gap_color}{gap:>4}{RESET}"
        )

    print("")
    print(f"{YELLOW}Datasets:{RESET}")
    result = kaggle_cmd("datasets", "list", "-m", check=False, capture_output=True)
    if result.returncode == 0:
        lines = result.stdout.splitlines()[2:]
        if lines:
            print("\n".join(lines))
    return 0


def cmd_link_competition(args: list[str]) -> int:
    if len(args) < 2:
        raise SystemExit("Usage: ./manage.sh link-competition <notebook-dir> <competition-slug>")
    directory, slug = args[0], args[1]
    path = ROOT / directory
    meta = path / "kernel-metadata.json"
    if not meta.exists():
        raise SystemExit(f"Error: kernel-metadata.json not found in {directory}")
    print(f"Linking {directory} -> competition '{slug}' ...")
    payload = json.loads(meta.read_text(encoding="utf-8"))
    sources = payload.get("competition_sources", [])
    if slug not in sources:
        sources.append(slug)
    payload["competition_sources"] = sources
    meta.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"competition_sources = {json.dumps(sources)}")
    if validate_for_push([str(path)], enforce_id_baseline=True) != 0:
        print(f"{RED}Fix validation errors before pushing.{RESET}")
        return 1
    print("Pushing updated notebook ...")
    rc = kaggle_cmd("kernels", "push", "-p", str(path), check=False).returncode
    if rc == 0:
        print(
            f"{GREEN}Done. Remember to accept the competition rules on Kaggle.com if push fails.{RESET}"
        )
    return rc


def cmd_competitions(_: list[str]) -> int:
    print(f"{BLUE}=== Active Medal-Eligible Competitions ==={RESET}")
    kaggle_cmd("competitions", "list", "--sort-by", "latestDeadline", "--category", "featured", check=False)
    print("")
    kaggle_cmd("competitions", "list", "--sort-by", "latestDeadline", "--category", "research", check=False)
    print("")
    kaggle_cmd("competitions", "list", "--sort-by", "latestDeadline", "--category", "playground", check=False)
    return 0


def cmd_dataset_ui_sync(args: list[str]) -> int:
    return run_script(PI_SCRIPTS / "dataset_metadata_sync.py", args)


def cmd_upload_covers(args: list[str]) -> int:
    return run_script(PI_SCRIPTS / "cover_image_upload.py", args)


def cmd_follow_users(args: list[str]) -> int:
    return run_script(PI_SCRIPTS / "follow_users.py", args)


def cmd_upvote(args: list[str]) -> int:
    return run_script(PI_SCRIPTS / "upvote_content.py", args)


def cmd_post_comment(args: list[str]) -> int:
    return run_script(PI_SCRIPTS / "comment_thread.py", args)


def cmd_draft_ops(args: list[str]) -> int:
    return run_module("kaggle_portfolio.ops.discussion_scheduler", ["--ops-report", *args])


def cmd_draft_set(args: list[str]) -> int:
    if not args:
        raise SystemExit(
            "Usage: ./manage.sh draft-set <draft_id> [--status ...] [--priority ...] "
            "[--deadline YYYY-MM-DD|--clear-deadline] [--schedule-weeks N]"
        )
    return run_module("kaggle_portfolio.ops.discussion_scheduler", ["--set-id", args[0], *args[1:]])


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    handler: Callable[[list[str]], int]
    args: str = ""
    requires_kaggle: bool = False
    hidden: bool = False


COMMANDS = [
    Command("status", "Show notebooks/datasets and Kaggle account status", cmd_status, requires_kaggle=True),
    Command("push-all", "Push all notebooks and datasets", cmd_push_all, requires_kaggle=True),
    Command("push-nb", "Push all notebooks", cmd_push_nb, requires_kaggle=True),
    Command("push-ds", "Push all datasets", cmd_push_ds, requires_kaggle=True),
    Command("push", "Push a specific notebook/dataset directory", cmd_push, "<dir>", True),
    Command("validate", "Validate kernel-metadata.json and dataset-metadata.json files", cmd_validate, "[dir]"),
    Command("votes", "Show vote counts with bronze/silver/gold medal threshold dashboard", cmd_votes, requires_kaggle=True),
    Command("competitions", "List active medal-eligible competitions", cmd_competitions, requires_kaggle=True),
    Command("link-competition", "Add competition_sources to a notebook and re-push", cmd_link_competition, "<dir> <slug>", True),
    Command("scorecard", "Generate medal operations scorecard report", lambda a: run_module("kaggle_portfolio.ops.medal_ops", ["scorecard", *a])),
    Command("badge-plan", "Generate ordered Kaggle badge roadmap report", lambda a: run_module("kaggle_portfolio.ops.medal_ops", ["badge-plan", *a])),
    Command("weekly-plan", "Generate weekly execution plan report", lambda a: run_module("kaggle_portfolio.ops.medal_ops", ["weekly-plan", *a])),
    Command("pace", "Generate medal progress pace analysis report", lambda a: run_module("kaggle_portfolio.ops.medal_ops", ["pace", *a])),
    Command("digest", "Print a one-message daily Grandmaster digest", lambda a: run_module("kaggle_portfolio.ops.medal_ops", ["digest", *a])),
    Command("sync", "Sync tracker metrics from live Kaggle CLI data", lambda a: run_module("kaggle_portfolio.ops.medal_ops", ["sync", *a])),
    Command("sync-template", "Generate CSV templates + export helper for offline sync", lambda a: run_module("kaggle_portfolio.ops.medal_ops", ["sync-template", *a])),
    Command("doctor", "Run preflight checks (tracker, sync inputs, environment)", lambda a: run_module("kaggle_portfolio.ops.medal_ops", ["doctor", *a])),
    Command("preflight", "Run the core repo gates: validate, doctor, quality, usability, draft SLA, tests", lambda a: run_module("kaggle_portfolio.ops.repo_ops", ["preflight", *a])),
    Command("quality", "Score notebook quality against rubric", lambda a: run_module("kaggle_portfolio.quality.notebook_quality", a)),
    Command("dataset-usability", "Score dataset usability and generate reports", lambda a: run_module("kaggle_portfolio.datasets.dataset_usability", a)),
    Command("usability-tracker", "Daily live tracker with threshold alerts and ranked action queue", lambda a: run_module("kaggle_portfolio.datasets.dataset_usability", ["--live", "--daily-tracker", "--alert-under", "0.8", "--target-rating", "1.0", "--fail-on-live-alert", "--write-live-ratings-csv", str(ROOT / "medal_ops" / "reports" / "latest-live-ratings.csv"), "--fallback-live-ratings-csv", str(ROOT / "medal_ops" / "reports" / "latest-live-ratings.csv"), *a])),
    Command("campaign-pack", "Generate multi-channel promotion campaign pack + queue", lambda a: run_module("kaggle_portfolio.campaigns.campaign_pack", a)),
    Command("campaign-run", "Execute campaign queue (show/claim/complete + runbook export)", lambda a: run_module("kaggle_portfolio.campaigns.campaign_dispatcher", a)),
    Command("campaign-execute", "Execute due campaign queue actions by posting discussion topics", lambda a: run_module("kaggle_portfolio.campaigns.campaign_execute", a), "[--limit N] [--dry-run] [--headed] [--channel NAME]"),
    Command("usability-benchmark", "Benchmark local datasets against public high-usability exemplars", lambda a: run_module("kaggle_portfolio.datasets.dataset_usability_benchmark", a), requires_kaggle=True),
    Command("publish-datasets", "Publish datasets through draft/live + quality gates", lambda a: run_module("kaggle_portfolio.datasets.dataset_publish_pipeline", a), "[--apply] [--all] [--min-score N] [--owner OWNER] [--max-items N]", True),
    Command("auth-doctor", "Validate Kaggle credentials, owner alignment, and upload auth", lambda a: run_module("kaggle_portfolio.ops.kaggle_auth_doctor", a)),
    Command("build-all", "Build all notebooks with build_notebook.py scripts", lambda a: run_module("kaggle_portfolio.notebooks.notebook_pipeline", a), "[--stale-only] [--push] [--validate-only] [--min-score N]"),
    Command("optimize-datasets", "Generate README.md + improve dataset descriptions", lambda a: run_module("kaggle_portfolio.datasets.dataset_optimizer", a), "[--push]"),
    Command("vote-plan", "Rank datasets by distance-to-medal + discoverability gaps", lambda a: run_module("kaggle_portfolio.datasets.dataset_vote_planner", a), "[--owner OWNER] [--json]", requires_kaggle=True),
    Command("post-discussion", "Post next queued discussion draft or rebuild queue window", lambda a: run_module("kaggle_portfolio.ops.discussion_scheduler", a), "[--dry-run|--init|--schedule-weeks N]"),
    Command("draft-ops", "Show draft backlog stage counts + priority queue", cmd_draft_ops),
    Command("draft-set", "Update draft metadata and rebalance queue schedule window", cmd_draft_set, "<id> [--status STATUS] [--priority PRIORITY] [--deadline YYYY-MM-DD|--clear-deadline] [--schedule-weeks N]"),
    Command("next-post", "Show the next ready discussion draft to post manually (safe assist)", lambda a: run_module("kaggle_portfolio.ops.discussion_scheduler", ["--next-post", *a])),
    Command("dataset-ui-sync", "Sync Kaggle UI-only dataset sections", cmd_dataset_ui_sync, "[--apply] [--headed] [--dataset <dir>] [--dataset-ref <owner/slug>]"),
    Command("promote-notebooks", "Generate notebook promotion plan for competition forums", lambda a: run_module("kaggle_portfolio.notebooks.notebook_promoter", a), "[--auto]"),
    Command("scout", "Scout active competitions ranked by medal opportunity", lambda a: run_module("kaggle_portfolio.notebooks.competition_scout", a), "[--update]"),
    Command("flywheel-status", "Print the growth-flywheel Reach-Score dashboard", lambda a: run_module("kaggle_portfolio.growth.flywheel", ["status", *a])),
    Command("flywheel-tick", "Run one growth-flywheel tick: score, gate, dispatch top safe actions", lambda a: run_module("kaggle_portfolio.growth.flywheel", ["tick", *a]), "[--dry-run]", True),
    Command("stale-content", "Detect stale notebooks, datasets, and outdated library versions", lambda a: run_module("kaggle_portfolio.ops.stale_content_detector", a), "[--max-nb-age N] [--max-ds-age N]"),
    Command("build-explore-notebooks", "Generate rich EDA explore notebooks for all datasets", lambda a: run_module("kaggle_portfolio.datasets.dataset_explore_generator", ["--all", *a]), "[--push]"),
    Command("create-competition-entry", "Scaffold a new competition entry from a competition slug", lambda a: run_module("kaggle_portfolio.notebooks.competition_entry", a), "<slug> [--gpu] [--push]"),
    Command("competition-lab", "Benchmark local competition models and optionally submit from the CLI", lambda a: run_module("kaggle_portfolio.notebooks.local_competition_lab", a), "<slug> [--write-submission] [--submit] [--force-download]"),
    Command("metadata-tracker", "Track metadata changes vs vote deltas over time", lambda a: run_module("kaggle_portfolio.ops.metadata_tracker", a), "<snapshot|annotate|report> [args...]"),
    Command("leaderboard", "Record/report competition leaderboard rank history", lambda a: run_module("kaggle_portfolio.ops.leaderboard_tracker", a), "<record|report> [--dry-run] [--json]", requires_kaggle=True),
    Command("smoke-live", "Safely exercise live Kaggle publish/post prerequisites without mutating Kaggle state", lambda a: run_module("kaggle_portfolio.ops.repo_ops", ["smoke-live", *a]), "[--owner OWNER] [--check-discussion-login]"),
    Command("upload-covers", "Upload cover images to Kaggle datasets via Playwright", cmd_upload_covers),
    Command("follow-users", "Follow Kaggle users to build visibility via Playwright", cmd_follow_users),
    Command("upvote", "Upvote Kaggle content via Playwright", cmd_upvote),
    Command("post-comment", "Post comments on Kaggle threads via Playwright", cmd_post_comment),
]

COMMAND_INDEX = {command.name: command for command in COMMANDS}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        command_name = "status"
    else:
        command_name = argv.pop(0)
    if command_name in {"help", "-h", "--help"}:
        print_usage()
        return 0
    command = COMMAND_INDEX.get(command_name)
    if command is None:
        print(f"Unknown command: {command_name}")
        print_usage()
        return 1
    if command.requires_kaggle:
        ensure_kaggle_ready()
    return command.handler(argv)
