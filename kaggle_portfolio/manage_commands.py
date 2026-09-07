from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import KaggleError
from kaggle_portfolio.shared.layout import METADATA_NAMES, RepoLayout
from kaggle_portfolio.shared.errors import CommandError

_DEPS: Deps | None = None


def deps() -> Deps:
    """The process-wide dependencies, built on first use.

    Lazily, not at import: constructing these used to mean two repo-wide rglob
    walks every time anything imported this module — including notebook_quality,
    which only wanted a twelve-line path predicate.
    """
    global _DEPS
    if _DEPS is None:
        _DEPS = Deps.resolve()
    return _DEPS


def set_deps(new: Deps | None) -> None:
    """Override or reset the process-wide dependencies. For tests and the CLI edge."""
    global _DEPS
    _DEPS = new


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PI_SCRIPTS = PACKAGE_ROOT / "pi-automation" / "scripts"
DEFAULT_CREDENTIALS = Path.home() / ".kaggle" / "kaggle.json"
# Kaggle rejects dataset uploads carrying more than this many keywords with
# "You have exceeded the max category limit". Measured 2026-08-19: 7 fails, 6 succeeds.
MAX_KEYWORDS = 6
SUSPICIOUS_PATTERN = re.compile(
    r"(password|secret|api_key|kgat_|kaggle_token)", re.IGNORECASE
)


def is_skipped(path: Path, root: Path | None = None) -> bool:
    """True when path lies inside a skipped directory, judged relative to root."""
    layout = deps().layout if root is None else RepoLayout.resolve(root)
    return layout.is_skipped(path)


TRUTHY = {"1", "true", "yes", "on"}

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
RESET = "\033[0m"


def discover_notebook_dirs() -> list[str]:
    return deps().layout.notebook_dirs()


def discover_dataset_dirs() -> list[str]:
    return deps().layout.dataset_dirs()


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
    return deps().client.available()


def require_kaggle_cli() -> None:
    if not has_kaggle_cli():
        raise CommandError(
            "Error: kaggle CLI not found. Install it with: pip install kaggle"
        )


def has_kaggle_credentials() -> tuple[bool, list[str]]:
    """Whether Kaggle credentials are resolvable, and where they came from."""
    state = deps().client.credentials()
    return bool(state), list(state.sources)


def require_kaggle_credentials() -> None:
    ok, _sources = has_kaggle_credentials()
    if ok:
        return
    raise CommandError(
        "Error: Kaggle credentials not found.\n"
        f"Create {DEFAULT_CREDENTIALS} (recommended) and run:\n"
        f"  chmod 600 {DEFAULT_CREDENTIALS}"
    )


def ensure_kaggle_ready() -> None:
    require_kaggle_cli()
    require_kaggle_credentials()


def run_script(path: Path, args: list[str]) -> int:
    result = subprocess.run(
        [sys.executable, str(path), *args], cwd=PACKAGE_ROOT, check=False
    )
    return result.returncode


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(deps().layout.root))
    except ValueError:
        return str(path)


def git_run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(deps().layout.root), *args],
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


def head_payload(path: Path, *, enforce_id_baseline: bool = False) -> dict | None:
    if not enforce_id_baseline or env_truthy("MANAGE_ALLOW_ID_CHANGE"):
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

    direct = (deps().layout.root / target).resolve()
    if direct.exists():
        return direct

    matches: list[Path] = []
    for rel in discover_notebook_dirs() + discover_dataset_dirs():
        rel_path = Path(rel)
        if str(rel_path) == target or rel_path.name == target:
            matches.append((deps().layout.root / rel_path).resolve())

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        joined = ", ".join(
            str(match.relative_to(deps().layout.root)) for match in matches[:10]
        )
        raise CommandError(f"Ambiguous target '{target}'. Matches: {joined}")
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
    for path in deps().layout.root.rglob("*-metadata.json"):
        if path.name not in METADATA_NAMES:
            continue
        if is_skipped(path):
            continue
        if in_scope(path, scope):
            files.append(path)
    return sorted(files)


def validate_kernel(
    path: Path, payload: dict, raw_text: str, *, enforce_id_baseline: bool = False
) -> list[str]:
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
        baseline = head_payload(path, enforce_id_baseline=enforce_id_baseline)
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
    required = (
        "id",
        "title",
        "licenses",
        "resources",
        "authors",
        "coverage",
        "provenance",
    )
    for field in required:
        if field not in payload:
            errors.append(f"missing '{field}'")

    keywords = payload.get("keywords")
    if isinstance(keywords, list) and len(keywords) > MAX_KEYWORDS:
        errors.append(
            f"{len(keywords)} keywords exceeds Kaggle's limit of {MAX_KEYWORDS}; "
            "the upload is rejected with 'You have exceeded the max category limit'"
        )

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
        for field in (
            "temporal_start_date",
            "temporal_end_date",
            "geospatial_coverage",
        ):
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


def cmd_validate(args: list[str], *, enforce_id_baseline: bool = False) -> int:
    """Validate metadata files.

    ``--enforce-id-baseline`` pins a tracked kernel's ``id`` to its committed
    value. It used to be reachable only by exporting VALIDATE_ENFORCE_ID_BASELINE,
    which made it an undocumented side channel; it is a flag now.
    """
    if "--enforce-id-baseline" in args:
        args = [a for a in args if a != "--enforce-id-baseline"]
        enforce_id_baseline = True
    scope = None
    if args:
        scope = resolve_target(args[0])
        if not scope.exists():
            print(
                f"{RED}Error:{RESET} validation target not found: {args[0]}",
                file=sys.stderr,
            )
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
            validate_kernel(meta, payload, raw, enforce_id_baseline=enforce_id_baseline)
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
    """Validate before a push, optionally pinning the kernel id to its HEAD baseline.

    This was signalled by setting an environment variable and restoring it in a
    `finally`. Correct only as long as that `finally` always ran — and in a
    long-lived interpreter shared by several commands, a leak would silently
    change how the next command validates. It is a parameter now.
    """
    return cmd_validate(args, enforce_id_baseline=enforce_id_baseline)


def cmd_status(_: list[str]) -> int:
    print(f"{BLUE}=== Kaggle Portfolio Status ==={RESET}")
    print("")
    print(f"{YELLOW}Notebooks:{RESET}")
    try:
        for kernel in deps().client.my_kernels()[:30]:
            print(f"  {kernel.ref:<60} {kernel.total_votes:>5} votes")
    except KaggleError as exc:
        print(f"  {RED}unavailable{RESET}: {exc}")
    print("")
    print(f"{YELLOW}Datasets:{RESET}")
    try:
        for dataset in deps().client.my_datasets()[:20]:
            print(f"  {dataset.ref:<60} {dataset.vote_count:>5} votes")
    except KaggleError as exc:
        print(f"  {RED}unavailable{RESET}: {exc}")
    print("")
    print(f"{YELLOW}Local directories:{RESET}")
    print(f"  Notebooks: {len(discover_notebook_dirs())}")
    print(f"  Datasets:  {len(discover_dataset_dirs())}")
    return 0


def push_dataset(path: Path) -> int:
    return 0 if deps().client.publish_dataset(path, "Updated content").ok else 1


def cmd_push(args: list[str]) -> int:
    if not args:
        raise CommandError("Usage: ./manage.sh push <directory>")
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
        return 0 if deps().client.push_kernel(path).ok else 1
    raise CommandError(f"Error: No metadata found in {path}")


def cmd_push_nb(_: list[str]) -> int:
    print(f"{BLUE}=== Pushing All Notebooks ==={RESET}")
    print(f"{YELLOW}Running pre-push validation...{RESET}")
    if validate_for_push([], enforce_id_baseline=True) != 0:
        print(f"{RED}Fix validation errors before pushing.{RESET}")
        return 1
    print("")
    success = 0
    failed = 0
    for rel in discover_notebook_dirs():
        path = deps().layout.root / rel
        if not (path / "kernel-metadata.json").exists():
            print(f"  {YELLOW}SKIP{RESET} {rel} (no kernel-metadata.json)")
            continue
        print(f"  Pushing {rel}... ", end="", flush=True)
        outcome = deps().client.push_kernel(path)
        if outcome.ok:
            print(f"{GREEN}OK{RESET}")
            success += 1
        else:
            print(f"{RED}FAILED{RESET}: {outcome.detail}")
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
    for rel in discover_dataset_dirs():
        path = deps().layout.root / rel
        if not (path / "dataset-metadata.json").exists():
            print(f"  {YELLOW}SKIP{RESET} {rel} (no dataset-metadata.json)")
            continue
        print(f"  Pushing {rel}... ", end="", flush=True)
        outcome = deps().client.publish_dataset(path, "Updated content")
        if outcome.ok:
            print(f"{GREEN}UPDATED{RESET}")
            success += 1
        else:
            print(f"{RED}FAILED{RESET}: {outcome.detail}")
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
    try:
        kernels = deps().client.my_kernels(kernel_type="notebook")
    except KaggleError as exc:
        print(f"{RED}Failed to fetch kernel list from Kaggle.{RESET} {exc}")
        return 1

    if not kernels:
        print("No kernels found.")
        return 0

    kernels = [k for k in kernels if k.ref and not k.is_private_placeholder]
    if not kernels:
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

    def sort_key(kernel) -> tuple[int, int]:
        _, threshold = next_tier(kernel.total_votes)
        return (-kernel.total_votes, threshold - kernel.total_votes)

    for kernel in sorted(kernels, key=sort_key):
        votes = kernel.total_votes
        tier_name, threshold = next_tier(votes)
        current = (
            "GOLD"
            if votes >= gold_t
            else "Silver"
            if votes >= silver_t
            else "Bronze"
            if votes >= bronze_t
            else "-"
        )
        gap = max(0, threshold - votes)
        gap_color = GREEN if 0 < gap <= 3 else YELLOW if 0 < gap <= 10 else RESET
        name = kernel.slug[:44]
        print(
            f"{medal_color(votes)}{name:<45}{RESET} {votes:>6}  {current:<8} "
            f"{tier_name:<8} {gap_color}{gap:>4}{RESET}"
        )

    print("")
    print(f"{YELLOW}Datasets:{RESET}")
    try:
        for dataset in deps().client.my_datasets():
            print(f"  {dataset.ref:<60} {dataset.vote_count:>5} votes")
    except KaggleError as exc:
        print(f"  {RED}unavailable{RESET}: {exc}")
    return 0


def cmd_link_competition(args: list[str]) -> int:
    if len(args) < 2:
        raise CommandError(
            "Usage: ./manage.sh link-competition <notebook-dir> <competition-slug>"
        )
    directory, slug = args[0], args[1]
    path = deps().layout.root / directory
    meta = path / "kernel-metadata.json"
    if not meta.exists():
        raise CommandError(f"Error: kernel-metadata.json not found in {directory}")
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
    rc = 0 if deps().client.push_kernel(path).ok else 1
    if rc == 0:
        print(
            f"{GREEN}Done. Remember to accept the competition rules on Kaggle.com if push fails.{RESET}"
        )
    return rc


def cmd_competitions(_: list[str]) -> int:
    print(f"{BLUE}=== Active Medal-Eligible Competitions ==={RESET}")
    failures = 0
    for category in ("featured", "research", "playground"):
        print(f"{YELLOW}{category.title()}:{RESET}")
        try:
            for comp in deps().client.search_competitions(category=category):
                print(f"  {comp.slug:<55} {comp.team_count:>6} teams  {comp.deadline}")
        except KaggleError as exc:
            # This used to discard the return code entirely and always report
            # success, so a broken CLI looked like an empty competition list.
            print(f"  {RED}unavailable{RESET}: {exc}")
            failures += 1
        print("")
    return 1 if failures else 0


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


def _scheduler_main(argv: list[str]) -> int:
    from kaggle_portfolio.ops import discussion_scheduler

    return discussion_scheduler.main(argv, deps=deps())


def cmd_draft_set(args: list[str]) -> int:
    """Set fields on one draft.

    Takes the draft id positionally and re-emits it as ``--set-id``, which is why
    this cannot be a plain declarative delegation.
    """
    if not args:
        raise CommandError(
            "Usage: ./manage.sh draft-set <draft_id> [--status ...] [--priority ...] "
            "[--deadline YYYY-MM-DD|--clear-deadline] [--schedule-weeks N]"
        )
    return _scheduler_main(["--set-id", args[0], *args[1:]])


def cmd_usability_tracker(args: list[str]) -> int:
    """The daily live usability tracker.

    The thresholds and the ratings-CSV path are policy, not user input, so they
    live here rather than in a lambda inside the command table. User arguments
    come last and win, which is why the CSV path is passed once per flag rather
    than relying on argparse seeing it twice.
    """
    from kaggle_portfolio.datasets import dataset_usability

    ratings_csv = str(deps().layout.reports_dir / "latest-live-ratings.csv")
    return dataset_usability.main(
        [
            "--live",
            "--daily-tracker",
            "--alert-under",
            "0.8",
            "--target-rating",
            "1.0",
            "--fail-on-live-alert",
            "--write-live-ratings-csv",
            ratings_csv,
            "--fallback-live-ratings-csv",
            ratings_csv,
            *args,
        ],
        deps=deps(),
    )


@dataclass(frozen=True)
class Command:
    """One CLI command, dispatched in-process wherever that is possible.

    Exactly one of ``handler``, ``module`` or ``script`` says how to run it:

    - ``handler`` — a function in this file.
    - ``module`` — a dotted path whose ``main(argv, deps=...)`` is called directly.
      The module is imported at DISPATCH time, not at registry construction: some
      command modules pull in pandas and sklearn, and one pulls in Playwright, so
      importing the table must not import the world.
    - ``script`` — a path run as a subprocess. Reserved for pi-automation, whose
      Playwright dependency must not become reachable from a kaggle_portfolio import.

    ``fixed_args`` are prepended to the user's argv, for modules whose CLI takes a
    subcommand (``medal_ops scorecard``) or a mode flag.
    """

    name: str
    description: str
    handler: Callable[[list[str]], int] | None = None
    args: str = ""
    requires_kaggle: bool = False
    hidden: bool = False
    module: str | None = None
    fixed_args: tuple[str, ...] = ()
    script: Path | None = None

    def run(self, argv: list[str], deps: Deps) -> int:
        if self.handler is not None:
            return self.handler(argv)
        if self.script is not None:
            return run_script(self.script, argv)
        if self.module is None:
            raise CommandError(
                f"Command {self.name!r} has no handler, module or script"
            )
        module = importlib.import_module(self.module)
        return module.main([*self.fixed_args, *argv], deps=deps)


COMMANDS = [
    Command(
        "status",
        "Show notebooks/datasets and Kaggle account status",
        cmd_status,
        requires_kaggle=True,
    ),
    Command(
        "push-all",
        "Push all notebooks and datasets",
        cmd_push_all,
        requires_kaggle=True,
    ),
    Command("push-nb", "Push all notebooks", cmd_push_nb, requires_kaggle=True),
    Command("push-ds", "Push all datasets", cmd_push_ds, requires_kaggle=True),
    Command(
        "push", "Push a specific notebook/dataset directory", cmd_push, "<dir>", True
    ),
    Command(
        "validate",
        "Validate kernel-metadata.json and dataset-metadata.json files",
        cmd_validate,
        "[dir]",
    ),
    Command(
        "votes",
        "Show vote counts with bronze/silver/gold medal threshold dashboard",
        cmd_votes,
        requires_kaggle=True,
    ),
    Command(
        "competitions",
        "List active medal-eligible competitions",
        cmd_competitions,
        requires_kaggle=True,
    ),
    Command(
        "link-competition",
        "Add competition_sources to a notebook and re-push",
        cmd_link_competition,
        "<dir> <slug>",
        requires_kaggle=True,
    ),
    Command(
        "scorecard",
        "Generate medal operations scorecard report",
        module="kaggle_portfolio.ops.medal_ops",
        fixed_args=("scorecard",),
    ),
    Command(
        "badge-plan",
        "Generate ordered Kaggle badge roadmap report",
        module="kaggle_portfolio.ops.medal_ops",
        fixed_args=("badge-plan",),
    ),
    Command(
        "weekly-plan",
        "Generate weekly execution plan report",
        module="kaggle_portfolio.ops.medal_ops",
        fixed_args=("weekly-plan",),
    ),
    Command(
        "pace",
        "Generate medal progress pace analysis report",
        module="kaggle_portfolio.ops.medal_ops",
        fixed_args=("pace",),
    ),
    Command(
        "digest",
        "Print a one-message daily Grandmaster digest",
        module="kaggle_portfolio.ops.medal_ops",
        fixed_args=("digest",),
    ),
    Command(
        "sync",
        "Sync tracker metrics from live Kaggle CLI data",
        module="kaggle_portfolio.ops.medal_ops",
        fixed_args=("sync",),
    ),
    Command(
        "sync-template",
        "Generate CSV templates + export helper for offline sync",
        module="kaggle_portfolio.ops.medal_ops",
        fixed_args=("sync-template",),
    ),
    Command(
        "doctor",
        "Run preflight checks (tracker, sync inputs, environment)",
        module="kaggle_portfolio.ops.medal_ops",
        fixed_args=("doctor",),
    ),
    Command(
        "preflight",
        "Run the core repo gates: validate, doctor, quality, usability, draft SLA, tests",
        module="kaggle_portfolio.ops.repo_ops",
        fixed_args=("preflight",),
    ),
    Command(
        "quality",
        "Score notebook quality against rubric",
        module="kaggle_portfolio.quality.notebook_quality",
    ),
    Command(
        "dataset-usability",
        "Score dataset usability and generate reports",
        module="kaggle_portfolio.datasets.dataset_usability",
    ),
    Command(
        "usability-tracker",
        "Daily live tracker with threshold alerts and ranked action queue",
        cmd_usability_tracker,
    ),
    Command(
        "campaign-pack",
        "Generate multi-channel promotion campaign pack + queue",
        module="kaggle_portfolio.campaigns.campaign_pack",
    ),
    Command(
        "campaign-run",
        "Execute campaign queue (show/claim/complete + runbook export)",
        module="kaggle_portfolio.campaigns.campaign_dispatcher",
    ),
    Command(
        "campaign-execute",
        "Execute due campaign queue actions by posting discussion topics",
        module="kaggle_portfolio.campaigns.campaign_execute",
        args="[--limit N] [--dry-run] [--headed] [--channel NAME]",
    ),
    Command(
        "usability-benchmark",
        "Benchmark local datasets against public high-usability exemplars",
        module="kaggle_portfolio.datasets.dataset_usability_benchmark",
        requires_kaggle=True,
    ),
    Command(
        "publish-datasets",
        "Publish datasets through draft/live + quality gates",
        module="kaggle_portfolio.datasets.dataset_publish_pipeline",
        args="[--apply] [--all] [--min-score N] [--owner OWNER] [--max-items N]",
        requires_kaggle=True,
    ),
    Command(
        "auth-doctor",
        "Validate Kaggle credentials, owner alignment, and upload auth",
        module="kaggle_portfolio.ops.kaggle_auth_doctor",
    ),
    Command(
        "build-all",
        "Build all notebooks with build_notebook.py scripts",
        module="kaggle_portfolio.notebooks.notebook_pipeline",
        args="[--stale-only] [--push] [--validate-only] [--min-score N]",
    ),
    Command(
        "optimize-datasets",
        "Generate README.md + improve dataset descriptions",
        module="kaggle_portfolio.datasets.dataset_optimizer",
        args="[--push]",
    ),
    Command(
        "vote-plan",
        "Rank datasets by distance-to-medal + discoverability gaps",
        module="kaggle_portfolio.datasets.dataset_vote_planner",
        args="[--owner OWNER] [--json]",
        requires_kaggle=True,
    ),
    Command(
        "post-discussion",
        "Post next queued discussion draft or rebuild queue window",
        module="kaggle_portfolio.ops.discussion_scheduler",
        args="[--dry-run|--init|--schedule-weeks N]",
    ),
    Command(
        "draft-ops",
        "Show draft backlog stage counts + priority queue",
        module="kaggle_portfolio.ops.discussion_scheduler",
        fixed_args=("--ops-report",),
    ),
    Command(
        "draft-set",
        "Update draft metadata and rebalance queue schedule window",
        cmd_draft_set,
        "<id> [--status STATUS] [--priority PRIORITY] [--deadline YYYY-MM-DD|--clear-deadline] [--schedule-weeks N]",
    ),
    Command(
        "next-post",
        "Show the next ready discussion draft to post manually (safe assist)",
        module="kaggle_portfolio.ops.discussion_scheduler",
        fixed_args=("--next-post",),
    ),
    Command(
        "dataset-ui-sync",
        "Sync Kaggle UI-only dataset sections",
        cmd_dataset_ui_sync,
        "[--apply] [--headed] [--dataset <dir>] [--dataset-ref <owner/slug>]",
    ),
    Command(
        "promote-notebooks",
        "Generate notebook promotion plan for competition forums",
        module="kaggle_portfolio.notebooks.notebook_promoter",
        args="[--auto]",
    ),
    Command(
        "scout",
        "Scout active competitions ranked by medal opportunity",
        module="kaggle_portfolio.notebooks.competition_scout",
        args="[--update]",
    ),
    Command(
        "flywheel-status",
        "Print the growth-flywheel Reach-Score dashboard",
        module="kaggle_portfolio.growth.flywheel",
        fixed_args=("status",),
    ),
    Command(
        "flywheel-tick",
        "Run one growth-flywheel tick: score, gate, dispatch top safe actions",
        module="kaggle_portfolio.growth.flywheel",
        fixed_args=("tick",),
        args="[--dry-run]",
        requires_kaggle=True,
    ),
    Command(
        "stale-content",
        "Detect stale notebooks, datasets, and outdated library versions",
        module="kaggle_portfolio.ops.stale_content_detector",
        args="[--max-nb-age N] [--max-ds-age N]",
    ),
    Command(
        "build-explore-notebooks",
        "Generate rich EDA explore notebooks for all datasets",
        module="kaggle_portfolio.datasets.dataset_explore_generator",
        fixed_args=("--all",),
        args="[--push]",
    ),
    Command(
        "create-competition-entry",
        "Scaffold a new competition entry from a competition slug",
        module="kaggle_portfolio.notebooks.competition_entry",
        args="<slug> [--gpu] [--push]",
    ),
    Command(
        "competition-lab",
        "Benchmark local competition models and optionally submit from the CLI",
        module="kaggle_portfolio.notebooks.local_competition_lab",
        args="<slug> [--write-submission] [--submit] [--force-download]",
    ),
    Command(
        "metadata-tracker",
        "Track metadata changes vs vote deltas over time",
        module="kaggle_portfolio.ops.metadata_tracker",
        args="<snapshot|annotate|report> [args...]",
    ),
    Command(
        "leaderboard",
        "Record/report competition leaderboard rank history",
        module="kaggle_portfolio.ops.leaderboard_tracker",
        args="<record|report> [--dry-run] [--json]",
        requires_kaggle=True,
    ),
    Command(
        "smoke-live",
        "Safely exercise live Kaggle publish/post prerequisites without mutating Kaggle state",
        module="kaggle_portfolio.ops.repo_ops",
        fixed_args=("smoke-live",),
        args="[--owner OWNER] [--check-discussion-login]",
    ),
    Command(
        "upload-covers",
        "Upload cover images to Kaggle datasets via Playwright",
        cmd_upload_covers,
    ),
    Command(
        "follow-users",
        "Follow Kaggle users to build visibility via Playwright",
        cmd_follow_users,
    ),
    Command("upvote", "Upvote Kaggle content via Playwright", cmd_upvote),
    Command(
        "post-comment",
        "Post comments on Kaggle threads via Playwright",
        cmd_post_comment,
    ),
]

COMMAND_INDEX = {command.name: command for command in COMMANDS}


def main(argv: list[str] | None = None, deps_override: Deps | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if deps_override is not None:
        set_deps(deps_override)
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
    try:
        if command.requires_kaggle:
            ensure_kaggle_ready()
        return command.run(argv, deps())
    except CommandError as exc:
        # Commands signal user-facing failure by raising, so that a failure deep
        # inside one does not terminate the interpreter the others share.
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return exc.exit_code
