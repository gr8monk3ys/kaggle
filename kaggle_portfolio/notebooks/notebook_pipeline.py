#!/usr/bin/env python3
"""Batch build + validate + optionally push all notebooks with build_notebook.py scripts.

Usage
-----
    # Build all notebooks and print quality scores
    python3 -m kaggle_portfolio.notebooks.notebook_pipeline

    # Build only notebooks where .ipynb is older than build_notebook.py
    python3 -m kaggle_portfolio.notebooks.notebook_pipeline --stale-only

    # Build + push notebooks scoring >= 60 (default threshold)
    python3 -m kaggle_portfolio.notebooks.notebook_pipeline --push

    # Validate metadata only, no build
    python3 -m kaggle_portfolio.notebooks.notebook_pipeline --validate-only

Invoked by manage.sh build-all.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from kaggle_portfolio.shared.kaggle_utils import kaggle_command

ROOT = Path(__file__).resolve().parents[2]

# Quality gate: notebooks below this score are not pushed automatically
DEFAULT_QUALITY_THRESHOLD = 60




def discover_build_scripts(root: Path) -> list[Path]:
    """Find all notebook build scripts under the repo root."""
    scripts = set(root.rglob("build_notebook.py"))
    scripts.update(root.rglob("_build_notebook.py"))
    return sorted(scripts)


def is_stale(script: Path) -> bool:
    """Return True if the .ipynb output is missing or older than the build script."""
    # Find the first .ipynb in the same directory
    ipynbs = list(script.parent.glob("*.ipynb"))
    if not ipynbs:
        return True
    newest_ipynb = max(ipynbs, key=lambda p: p.stat().st_mtime)
    return newest_ipynb.stat().st_mtime < script.stat().st_mtime


def build_notebook(script: Path) -> tuple[bool, str]:
    """Run a build_notebook.py script and return (success, output)."""
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    output = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0, output


def score_notebook_dir(nb_dir: Path) -> tuple[float, str]:
    """Run notebook quality scoring on a directory and return (score, summary)."""
    ipynbs = sorted(nb_dir.glob("*.ipynb"))
    if not ipynbs:
        return 0.0, "no .ipynb found"
    nb_path = ipynbs[0]
    try:
        from kaggle_portfolio.quality import notebook_quality

        # notebook_quality.score_notebook expects a root used for slug generation
        try:
            nb_path.relative_to(ROOT)
            score_root = ROOT
        except ValueError:
            score_root = nb_path.parent

        scored = notebook_quality.score_notebook(path=nb_path, root=score_root, min_score=0)
        summary = f"Score: {scored.score}/100"
        if scored.error:
            summary = f"{summary} ({scored.error})"
        return float(scored.score), summary
    except Exception as exc:
        return 0.0, f"scoring failed: {exc}"


def push_notebook(nb_dir: Path) -> tuple[bool, str]:
    """Push a notebook directory to Kaggle."""
    cli = kaggle_command()
    result = subprocess.run(
        [*cli, "kernels", "push", "-p", str(nb_dir)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def run_validate() -> bool:
    """Run manage.sh validate and return True if clean."""
    result = subprocess.run(
        ["bash", str(ROOT / "manage.sh"), "validate"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    print(result.stdout)
    if result.stderr.strip():
        print(result.stderr)
    return result.returncode == 0


GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
RESET = "\033[0m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch build + validate + push notebooks.")
    parser.add_argument("--stale-only", action="store_true",
                        help="Only build notebooks whose .ipynb is older than the build script.")
    parser.add_argument("--push", action="store_true",
                        help="Push notebooks that pass the quality threshold.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run metadata validation only, no build or push.")
    parser.add_argument("--min-score", type=int, default=DEFAULT_QUALITY_THRESHOLD,
                        help=f"Minimum quality score to push (default: {DEFAULT_QUALITY_THRESHOLD}).")
    args = parser.parse_args(argv)

    print(f"{BLUE}=== Notebook Pipeline ==={RESET}\n")

    if args.validate_only:
        print("Running metadata validation only...")
        ok = run_validate()
        return 0 if ok else 1

    scripts = discover_build_scripts(ROOT)
    if not scripts:
        print("No build_notebook.py files found.")
        return 0

    print(f"Found {len(scripts)} build script(s).\n")

    results: list[dict] = []

    for script in scripts:
        rel = script.parent.relative_to(ROOT)
        tag = str(rel)

        if args.stale_only and not is_stale(script):
            print(f"  {YELLOW}SKIP{RESET} {tag} (up-to-date)")
            continue

        print(f"  Building {tag}... ", end="", flush=True)
        ok, output = build_notebook(script)
        if ok:
            print(f"{GREEN}OK{RESET}")
            # First line of output has the path; rest has cell counts
            for line in output.splitlines()[1:]:
                print(f"    {line}")
        else:
            print(f"{RED}FAILED{RESET}")
            print(f"    {output[:200]}")
            results.append({"dir": script.parent, "tag": tag, "built": False, "score": 0.0})
            continue

        # Score the built notebook
        score, score_summary = score_notebook_dir(script.parent)
        score_color = GREEN if score >= args.min_score else (YELLOW if score >= 40 else RED)
        print(f"    Quality: {score_color}{score:.0f}/100{RESET}  {score_summary[:80]}")

        pushed = False
        if args.push:
            if score >= args.min_score:
                print(f"    Pushing {tag}... ", end="", flush=True)
                push_ok, push_out = push_notebook(script.parent)
                if push_ok:
                    print(f"{GREEN}pushed{RESET}")
                    pushed = True
                else:
                    print(f"{RED}push failed{RESET}: {push_out[:120]}")
            else:
                print(f"    {YELLOW}Skipping push{RESET} (score {score:.0f} < {args.min_score})")

        results.append({"dir": script.parent, "tag": tag, "built": True,
                        "score": score, "pushed": pushed})

    # Summary
    print(f"\n{BLUE}=== Summary ==={RESET}")
    built = sum(1 for r in results if r.get("built"))
    pushed_count = sum(1 for r in results if r.get("pushed"))
    failed = sum(1 for r in results if not r.get("built"))

    print(f"  Built:  {built}")
    print(f"  Failed: {failed}")
    if args.push:
        print(f"  Pushed: {pushed_count}")

    if results:
        print(f"\n{'Notebook':<40} {'Score':>6}  {'Status'}")
        print("-" * 60)
        for r in sorted(results, key=lambda x: -x.get("score", 0)):
            score_str = f"{r['score']:.0f}/100" if r.get("built") else "FAILED"
            status = "pushed" if r.get("pushed") else ("built" if r.get("built") else "FAILED")
            tag = r["tag"][:39]
            col = GREEN if r.get("pushed") else (YELLOW if r.get("built") else RED)
            print(f"  {col}{tag:<40}{RESET} {score_str:>6}  {status}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
