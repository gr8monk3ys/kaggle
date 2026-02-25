#!/usr/bin/env python3
"""Notebook quality scoring for Kaggle portfolio assets."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("medal_ops")

CRITERION_MAX = {
    "title_structure": 10,
    "content_depth": 10,
    "markdown_balance": 12,
    "section_coverage": 22,
    "visualization_signal": 12,
    "reproducibility_signal": 12,
    "insight_signal": 10,
    "closing_signal": 12,
}

CRITERION_HINT = {
    "title_structure": "Ensure first non-empty cell is a markdown H1 title.",
    "content_depth": "Add depth (more meaningful cells and analysis steps).",
    "markdown_balance": "Improve markdown/code balance for readability.",
    "section_coverage": "Strengthen core sections: objective, data, method, evaluation, conclusion.",
    "visualization_signal": "Add stronger visual analysis (plots/charts).",
    "reproducibility_signal": "Add explicit reproducibility controls (seeds/random_state).",
    "insight_signal": "Increase interpretation language (insights, trade-offs, limitations).",
    "closing_signal": "Add a clear closing summary with next steps.",
}

CRITERION_PLAYBOOK = {
    "title_structure": "Start with a markdown H1 that clearly states problem, data, and approach.",
    "content_depth": "Add at least 3-5 meaningful analysis/modeling steps with interpretation between them.",
    "markdown_balance": "Add explanatory markdown before/after dense code blocks to improve narrative flow.",
    "section_coverage": "Ensure objective, data, method, evaluation, and conclusion sections all exist.",
    "visualization_signal": "Add targeted charts for distributions, errors, or feature effects.",
    "reproducibility_signal": "Set deterministic seeds and explicitly log key config/hyperparameters.",
    "insight_signal": "Add explicit interpretation lines that explain why results changed.",
    "closing_signal": "End with concise takeaways and concrete next-step experiments.",
}

SECTION_PATTERNS = {
    "objective": [r"\b(objective|introduction|overview|problem statement|goal)\b"],
    "data": [r"\b(dataset|data overview|eda|exploratory data|feature(s)? overview)\b"],
    "method": [r"\b(method|approach|model(ing)?|training|pipeline|architecture)\b"],
    "evaluation": [r"\b(result(s)?|evaluation|metric(s)?|validation|leaderboard)\b"],
    "conclusion": [r"\b(conclusion|summary|takeaway(s)?|next step(s)?|final thoughts)\b"],
}

SECTION_PLAYBOOK = {
    "objective": "Add an objective section with task definition and success metric.",
    "data": "Add a data section with schema, quality checks, and quick EDA summary.",
    "method": "Add a method section that explains feature/model/training choices.",
    "evaluation": "Add an evaluation section with metric table and error analysis.",
    "conclusion": "Add a conclusion section with outcomes and next-step plan.",
}

VISUALIZATION_KEYWORDS = [
    "matplotlib",
    "plt.",
    "seaborn",
    "sns.",
    "plotly",
    "px.",
    "go.",
    "altair",
    "bokeh",
    "hvplot",
    ".plot(",
]

REPRODUCIBILITY_KEYWORDS = [
    "random_state",
    "np.random.seed",
    "torch.manual_seed",
    "tf.random.set_seed",
    "seed(",
    "deterministic",
]

INSIGHT_KEYWORDS = [
    "insight",
    "observation",
    "finding",
    "interpret",
    "because",
    "therefore",
    "trade-off",
    "limitation",
    "caveat",
    "hypothesis",
]

ACTION_KEYWORDS = [
    "next step",
    "future work",
    "recommend",
    "improve",
    "action item",
    "todo",
]

CLOSING_KEYWORDS = ["conclusion", "summary", "takeaway", "next step", "final thoughts"]


@dataclass(frozen=True)
class NotebookScore:
    notebook_path: Path
    slug: str
    title: str
    score: int
    passed: bool
    criteria: dict[str, int]
    missing: list[str]
    error: str | None = None


def parse_iso_date(text: str) -> date | None:
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def resolve_today(today_override: str | None) -> date:
    if today_override is None:
        return date.today()
    parsed = parse_iso_date(today_override)
    if not parsed:
        raise SystemExit(f"Invalid --today value: {today_override}")
    return parsed


def source_to_text(source: Any) -> str:
    if isinstance(source, list):
        return "".join(str(item) for item in source)
    if isinstance(source, str):
        return source
    return ""


def discover_notebooks(root: Path, scope: str) -> tuple[list[Path], list[str]]:
    notebooks: list[Path] = []
    warnings: list[str] = []

    for metadata_path in sorted(root.rglob("kernel-metadata.json")):
        if ".git" in metadata_path.parts or ".venv" in metadata_path.parts:
            continue
        rel_metadata = metadata_path.relative_to(root)
        if scope == "portfolio" and rel_metadata.parts and rel_metadata.parts[0] == "datasets":
            continue

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not parse {metadata_path}: {exc}")
            continue

        code_file = metadata.get("code_file")
        if not isinstance(code_file, str) or not code_file.strip():
            warnings.append(f"Missing `code_file` in {metadata_path}")
            continue

        notebook_path = metadata_path.parent / code_file
        if not notebook_path.exists():
            warnings.append(f"Notebook missing for {metadata_path}: {notebook_path}")
            continue
        if notebook_path.suffix.lower() != ".ipynb":
            warnings.append(f"Notebook is not .ipynb: {notebook_path}")
            continue
        notebooks.append(notebook_path)

    return notebooks, warnings


def contains_any(text: str, keywords: list[str]) -> bool:
    text_l = text.lower()
    return any(keyword in text_l for keyword in keywords)


def compute_section_coverage(markdown_text: str) -> tuple[int, list[str], int]:
    hits = 0
    missing_categories: list[str] = []
    for category, patterns in SECTION_PATTERNS.items():
        found = any(re.search(pattern, markdown_text, flags=re.IGNORECASE) for pattern in patterns)
        if found:
            hits += 1
        else:
            missing_categories.append(category)

    score = hits * 4
    if hits >= 4:
        score += 2
    return score, missing_categories, hits


def extract_title(cells: list[dict[str, Any]], fallback: str) -> str:
    for cell in cells:
        text = source_to_text(cell.get("source")).strip()
        if not text:
            continue
        if cell.get("cell_type") == "markdown":
            heading = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
            if heading:
                return heading.group(1).strip()
        break
    return fallback


def score_notebook(path: Path, root: Path, min_score: int) -> NotebookScore:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return NotebookScore(
            notebook_path=path,
            slug=str(path.relative_to(root)),
            title=path.stem,
            score=0,
            passed=False,
            criteria={key: 0 for key in CRITERION_MAX},
            missing=["Notebook is unreadable JSON."],
            error=str(exc),
        )

    cells = payload.get("cells", [])
    if not isinstance(cells, list):
        cells = []

    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]

    markdown_texts = [source_to_text(cell.get("source")) for cell in markdown_cells]
    code_texts = [source_to_text(cell.get("source")) for cell in code_cells]
    all_markdown_text = "\n".join(markdown_texts)
    all_code_text = "\n".join(code_texts)

    first_non_empty = None
    for cell in cells:
        text = source_to_text(cell.get("source")).strip()
        if text:
            first_non_empty = (cell, text)
            break

    criteria: dict[str, int] = {}

    title_ok = bool(
        first_non_empty
        and first_non_empty[0].get("cell_type") == "markdown"
        and re.search(r"^#\s+\S+", first_non_empty[1], flags=re.MULTILINE)
    )
    criteria["title_structure"] = 10 if title_ok else 0

    total_cells = len(cells)
    if total_cells >= 30:
        criteria["content_depth"] = 10
    elif total_cells >= 20:
        criteria["content_depth"] = 8
    elif total_cells >= 12:
        criteria["content_depth"] = 5
    elif total_cells >= 8:
        criteria["content_depth"] = 2
    else:
        criteria["content_depth"] = 0

    markdown_ratio = (len(markdown_cells) / total_cells) if total_cells else 0.0
    if 0.25 <= markdown_ratio <= 0.65:
        criteria["markdown_balance"] = 12
    elif 0.18 <= markdown_ratio <= 0.75:
        criteria["markdown_balance"] = 8
    elif 0.1 <= markdown_ratio <= 0.9:
        criteria["markdown_balance"] = 4
    else:
        criteria["markdown_balance"] = 0

    section_score, missing_sections, section_hits = compute_section_coverage(all_markdown_text)
    criteria["section_coverage"] = section_score

    if contains_any(all_code_text, VISUALIZATION_KEYWORDS):
        criteria["visualization_signal"] = 12
    elif "plot" in all_markdown_text.lower():
        criteria["visualization_signal"] = 6
    else:
        criteria["visualization_signal"] = 0

    if contains_any(all_code_text, REPRODUCIBILITY_KEYWORDS):
        criteria["reproducibility_signal"] = 12
    elif contains_any(all_markdown_text, REPRODUCIBILITY_KEYWORDS):
        criteria["reproducibility_signal"] = 6
    else:
        criteria["reproducibility_signal"] = 0

    insight_hits = sum(1 for word in INSIGHT_KEYWORDS if word in all_markdown_text.lower())
    if insight_hits >= 4:
        criteria["insight_signal"] = 10
    elif insight_hits >= 2:
        criteria["insight_signal"] = 6
    elif insight_hits >= 1:
        criteria["insight_signal"] = 3
    else:
        criteria["insight_signal"] = 0

    tail_count = max(2, int(len(markdown_texts) * 0.3)) if markdown_texts else 0
    tail_markdown = "\n".join(markdown_texts[-tail_count:]) if tail_count else ""
    has_closing = contains_any(tail_markdown, CLOSING_KEYWORDS)
    has_action = contains_any(tail_markdown, ACTION_KEYWORDS)
    if has_closing and has_action:
        criteria["closing_signal"] = 12
    elif has_closing:
        criteria["closing_signal"] = 8
    elif has_action:
        criteria["closing_signal"] = 4
    else:
        criteria["closing_signal"] = 0

    raw_score = sum(criteria.values())
    score = max(0, min(100, raw_score))

    missing: list[str] = []
    for key, max_points in CRITERION_MAX.items():
        if criteria[key] <= (max_points // 2):
            missing.append(CRITERION_HINT[key])
    if section_hits < 4:
        missing.append(f"Missing section categories: {', '.join(missing_sections)}.")
    if score >= min_score and len(missing) > 2:
        missing = missing[:2]

    title = extract_title(cells, path.stem)
    return NotebookScore(
        notebook_path=path,
        slug=str(path.relative_to(root)),
        title=title,
        score=score,
        passed=score >= min_score,
        criteria=criteria,
        missing=missing,
    )


def parse_missing_sections(missing: list[str]) -> list[str]:
    for message in missing:
        prefix = "Missing section categories:"
        if not message.startswith(prefix):
            continue
        raw = message[len(prefix) :].strip().rstrip(".")
        return [part.strip() for part in raw.split(",") if part.strip()]
    return []


def build_priority_actions(item: NotebookScore, top_n: int) -> list[dict[str, Any]]:
    if item.error:
        return [
            {
                "criterion": "notebook_readability",
                "impact": 100,
                "task": "Repair notebook JSON structure so quality scoring can run.",
            }
        ]

    missing_sections = parse_missing_sections(item.missing)
    actions: list[dict[str, Any]] = []
    for criterion, max_points in CRITERION_MAX.items():
        current = item.criteria.get(criterion, 0)
        gap = max_points - current
        if gap <= 0:
            continue
        task = CRITERION_PLAYBOOK[criterion]
        if criterion == "section_coverage" and missing_sections:
            section_tasks = [SECTION_PLAYBOOK[key] for key in missing_sections if key in SECTION_PLAYBOOK]
            if section_tasks:
                task = task + " " + " ".join(section_tasks[:3])
        actions.append({"criterion": criterion, "impact": gap, "task": task})

    actions.sort(key=lambda action: (-int(action["impact"]), str(action["criterion"])))
    return actions[: max(1, top_n)]


def aggregate_criterion_gaps(scores: list[NotebookScore]) -> list[tuple[str, int]]:
    totals: defaultdict[str, int] = defaultdict(int)
    for item in scores:
        if item.error:
            totals["notebook_readability"] += 100
            continue
        for criterion, max_points in CRITERION_MAX.items():
            gap = max_points - item.criteria.get(criterion, 0)
            if gap > 0:
                totals[criterion] += gap

    return sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))


def select_fix_candidates(
    scores: list[NotebookScore],
    target_score: int,
    max_notebooks: int,
) -> list[NotebookScore]:
    candidates = [item for item in scores if item.error or item.score < target_score]
    return sorted(candidates, key=lambda item: (item.score, item.slug))[: max(1, max_notebooks)]


def generate_fixer_markdown(
    scores: list[NotebookScore],
    today: date,
    target_score: int,
    top_actions: int,
    max_notebooks: int,
) -> str:
    candidates = select_fix_candidates(scores, target_score=target_score, max_notebooks=max_notebooks)
    gap_totals = aggregate_criterion_gaps(scores)

    lines = [
        "# Notebook Quality Fix Checklist",
        "",
        f"- Generated: {today.isoformat()}",
        f"- Fix target score: {target_score}",
        f"- Candidate notebooks: {len(candidates)}",
        "",
        "## Portfolio Priorities",
        "",
    ]

    if gap_totals:
        for criterion, total_gap in gap_totals[:8]:
            if criterion == "notebook_readability":
                task = "Repair unreadable notebook JSON files."
            else:
                task = CRITERION_PLAYBOOK[criterion]
            lines.append(f"- `{criterion}` (total gap {total_gap}): {task}")
    else:
        lines.append("- No remaining rubric gaps detected.")
    lines.append("")

    lines.append("## Notebook Checklists")
    lines.append("")
    if not candidates:
        lines.append(f"- No notebooks below target score {target_score}.")
        lines.append("")
        return "\n".join(lines)

    for item in candidates:
        score_label = f"{item.score}"
        if item.error:
            score_label = f"{item.score} (error)"
        lines.append(f"### `{item.slug}`")
        lines.append(f"- Current score: {score_label}")
        lines.append(f"- Target: {target_score}")
        actions = build_priority_actions(item, top_n=top_actions)
        if not actions:
            lines.append("- [ ] No action required.")
        else:
            for action in actions:
                lines.append(
                    f"- [ ] (+{action['impact']}) `{action['criterion']}`: {action['task']}"
                )
        lines.append("")

    return "\n".join(lines)


def build_fixer_json(
    scores: list[NotebookScore],
    today: date,
    target_score: int,
    top_actions: int,
    max_notebooks: int,
) -> dict[str, Any]:
    candidates = select_fix_candidates(scores, target_score=target_score, max_notebooks=max_notebooks)
    gap_totals = aggregate_criterion_gaps(scores)
    return {
        "generated_on": today.isoformat(),
        "target_score": target_score,
        "top_actions_per_notebook": top_actions,
        "max_notebooks": max_notebooks,
        "portfolio_priorities": [
            {"criterion": criterion, "total_gap": total_gap}
            for criterion, total_gap in gap_totals
        ],
        "candidates": [
            {
                "path": item.slug,
                "title": item.title,
                "score": item.score,
                "error": item.error,
                "actions": build_priority_actions(item, top_n=top_actions),
            }
            for item in candidates
        ],
    }


def generate_quality_markdown(
    scores: list[NotebookScore],
    warnings: list[str],
    min_score: int,
    scope: str,
    today: date,
) -> str:
    if not scores:
        return (
            "# Kaggle Notebook Quality Report\n\n"
            f"- Generated: {today.isoformat()}\n"
            "- No notebooks discovered.\n"
        )

    values = [item.score for item in scores]
    average_score = statistics.mean(values)
    median_score = statistics.median(values)
    passed = sum(1 for item in scores if item.passed)
    failed = len(scores) - passed

    lines = [
        "# Kaggle Notebook Quality Report",
        "",
        f"- Generated: {today.isoformat()}",
        f"- Scope: {scope}",
        f"- Notebook count: {len(scores)}",
        f"- Threshold: {min_score}",
        f"- Pass rate: {passed}/{len(scores)} ({(passed / len(scores)) * 100:.1f}%)",
        f"- Average score: {average_score:.1f}",
        f"- Median score: {median_score:.1f}",
        "",
        "## Rubric Weights",
        "",
        "| Criterion | Max Points |",
        "|---|---:|",
    ]
    for key, max_points in CRITERION_MAX.items():
        lines.append(f"| {key} | {max_points} |")
    lines.append("")

    lines.extend(
        [
            "## Notebook Scores",
            "",
            "| Notebook | Score | Status | Priority Fixes |",
            "|---|---:|---|---|",
        ]
    )

    for item in sorted(scores, key=lambda s: (s.score, s.slug)):
        status = "PASS" if item.passed else "IMPROVE"
        if item.error:
            fixes = f"Unreadable notebook JSON ({item.error})"
        elif item.missing:
            fixes = "; ".join(item.missing[:2])
        else:
            fixes = "Strong baseline"
        lines.append(f"| `{item.slug}` | {item.score} | {status} | {fixes} |")
    lines.append("")

    lines.append("## Focus Queue")
    lines.append("")
    if failed == 0:
        lines.append("- No notebooks are below threshold.")
    else:
        for item in sorted((s for s in scores if not s.passed), key=lambda s: s.score)[:10]:
            lines.append(f"- `{item.slug}` ({item.score}): {'; '.join(item.missing[:3])}")
    lines.append("")

    lines.append("## Discovery Warnings")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")

    return "\n".join(lines)


def build_json_report(
    scores: list[NotebookScore],
    warnings: list[str],
    min_score: int,
    scope: str,
    today: date,
) -> dict[str, Any]:
    return {
        "generated_on": today.isoformat(),
        "scope": scope,
        "min_score": min_score,
        "summary": {
            "count": len(scores),
            "passed": sum(1 for item in scores if item.passed),
            "failed": sum(1 for item in scores if not item.passed),
            "average_score": round(statistics.mean([item.score for item in scores]), 2) if scores else 0.0,
            "median_score": round(statistics.median([item.score for item in scores]), 2) if scores else 0.0,
        },
        "warnings": warnings,
        "notebooks": [
            {
                "path": item.slug,
                "title": item.title,
                "score": item.score,
                "status": "pass" if item.passed else "improve",
                "criteria": item.criteria,
                "missing": item.missing,
                "error": item.error,
            }
            for item in sorted(scores, key=lambda s: (s.score, s.slug))
        ],
    }


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Kaggle notebooks against a quality rubric.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Report output root.")
    parser.add_argument("--today", default=None, help="Override date in YYYY-MM-DD format.")
    parser.add_argument("--min-score", type=int, default=70, help="Minimum passing score (0-100).")
    parser.add_argument(
        "--scope",
        choices=("all", "portfolio"),
        default="all",
        help="Notebook scope: all kernels or only top-level portfolio notebooks.",
    )
    parser.add_argument(
        "--fail-under-threshold",
        action="store_true",
        help="Return exit code 1 if any notebook score is below --min-score.",
    )
    parser.add_argument(
        "--fix-target-score",
        type=int,
        default=85,
        help="Generate checklists for notebooks below this score.",
    )
    parser.add_argument(
        "--fix-top-actions",
        type=int,
        default=4,
        help="Maximum prioritized actions per notebook in fixer checklist.",
    )
    parser.add_argument(
        "--fix-max-notebooks",
        type=int,
        default=12,
        help="Maximum notebooks included in fixer checklist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_score < 0 or args.min_score > 100:
        raise SystemExit("--min-score must be between 0 and 100")
    if args.fix_target_score < 0 or args.fix_target_score > 100:
        raise SystemExit("--fix-target-score must be between 0 and 100")
    if args.fix_top_actions <= 0:
        raise SystemExit("--fix-top-actions must be >= 1")
    if args.fix_max_notebooks <= 0:
        raise SystemExit("--fix-max-notebooks must be >= 1")

    today = resolve_today(args.today)
    root = Path(args.root).resolve()
    output_root = Path(args.output_root)

    notebooks, warnings = discover_notebooks(root, scope=args.scope)
    if not notebooks:
        raise SystemExit("No notebooks discovered from kernel-metadata.json files.")

    scores = [score_notebook(path=notebook, root=root, min_score=args.min_score) for notebook in notebooks]

    markdown = generate_quality_markdown(scores, warnings, args.min_score, args.scope, today)
    json_report = build_json_report(scores, warnings, args.min_score, args.scope, today)
    fixer_markdown = generate_fixer_markdown(
        scores,
        today=today,
        target_score=args.fix_target_score,
        top_actions=args.fix_top_actions,
        max_notebooks=args.fix_max_notebooks,
    )
    fixer_json = build_fixer_json(
        scores,
        today=today,
        target_score=args.fix_target_score,
        top_actions=args.fix_top_actions,
        max_notebooks=args.fix_max_notebooks,
    )

    reports_dir = output_root / "reports"
    dated_md_path = reports_dir / f"notebook-quality-{today.isoformat()}.md"
    latest_md_path = reports_dir / "latest-notebook-quality.md"
    dated_json_path = reports_dir / f"notebook-quality-{today.isoformat()}.json"
    latest_json_path = reports_dir / "latest-notebook-quality.json"
    dated_fixer_md_path = reports_dir / f"notebook-quality-fixes-{today.isoformat()}.md"
    latest_fixer_md_path = reports_dir / "latest-notebook-quality-fixes.md"
    dated_fixer_json_path = reports_dir / f"notebook-quality-fixes-{today.isoformat()}.json"
    latest_fixer_json_path = reports_dir / "latest-notebook-quality-fixes.json"

    write_text(dated_md_path, markdown)
    write_text(latest_md_path, markdown)
    write_json(dated_json_path, json_report)
    write_json(latest_json_path, json_report)
    write_text(dated_fixer_md_path, fixer_markdown)
    write_text(latest_fixer_md_path, fixer_markdown)
    write_json(dated_fixer_json_path, fixer_json)
    write_json(latest_fixer_json_path, fixer_json)

    failed = [item for item in scores if not item.passed]
    print(f"Notebook quality report written: {dated_md_path}")
    print(f"Latest notebook quality report: {latest_md_path}")
    print(f"Notebook fixer checklist written: {dated_fixer_md_path}")
    print(f"Latest notebook fixer checklist: {latest_fixer_md_path}")
    print(
        "Summary: "
        f"{len(scores) - len(failed)} pass, {len(failed)} improve, "
        f"average {statistics.mean([item.score for item in scores]):.1f}"
    )

    if args.fail_under_threshold and failed:
        print(f"Quality gate failed: {len(failed)} notebook(s) below threshold {args.min_score}.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
