#!/usr/bin/env python3
"""Sync Kaggle dataset metadata card fields via browser automation.

This script reads local `datasets/*/dataset-metadata.json` files and updates
the corresponding Kaggle dataset UI sections that are not exposed by the CLI
metadata model (authors, coverage, DOI, provenance, citations).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "datasets"
DEFAULT_STORAGE_STATE = REPO_ROOT / "pi-automation" / "data" / "kaggle_storage_state.json"
DEFAULT_TIMEOUT_MS = 20000
SECTION_TITLES = ("Authors", "Coverage", "DOI Citation", "Provenance", "Citations")


@dataclass(frozen=True)
class DatasetUiPayload:
    dataset_dir: str
    dataset_ref: str
    author_name: str
    author_bio: str
    temporal_start_date: str
    temporal_end_date: str
    geospatial_coverage: str
    doi: str
    sources: list[str]
    collection_methodology: str
    citations: list[str]


@dataclass(frozen=True)
class SectionResult:
    name: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class DatasetResult:
    dataset_ref: str
    editor_url: str
    sections: list[SectionResult]


def load_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def coerce_list_of_text(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    values: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            values.append(text)
    return values


def default_citation(meta: dict, dataset_ref: str) -> str:
    title = str(meta.get("title", "Dataset")).strip() or "Dataset"
    year = date.today().year
    return f"Scaturchio, Lorenzo ({year}). {title}. Kaggle Dataset. https://www.kaggle.com/datasets/{dataset_ref}"


def build_payload(meta: dict, dataset_dir: str, *, force_doi: str | None = None) -> DatasetUiPayload:
    dataset_ref = str(meta.get("id", "")).strip().lower()
    if not dataset_ref or "/" not in dataset_ref:
        raise ValueError(f"{dataset_dir}: metadata id must be owner/slug")

    authors = meta.get("authors") if isinstance(meta.get("authors"), list) else []
    first_author = authors[0] if authors and isinstance(authors[0], dict) else {}
    author_name = str(first_author.get("name", "")).strip() or "Lorenzo Scaturchio"
    author_bio = (
        str(first_author.get("bio", "")).strip()
        or "Independent ML engineer building synthetic, education-first datasets."
    )

    coverage = meta.get("coverage") if isinstance(meta.get("coverage"), dict) else {}
    temporal_start_date = str(coverage.get("temporal_start_date", "")).strip()
    temporal_end_date = str(coverage.get("temporal_end_date", "")).strip()
    geospatial_coverage = str(coverage.get("geospatial_coverage", "")).strip() or "Global (synthetic)"

    doi = (force_doi or str(meta.get("doi", "")).strip()).strip()
    if doi.lower() in {"not assigned", "none", "n/a", "-"}:
        doi = ""

    provenance = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}
    sources = coerce_list_of_text(provenance.get("sources"))
    collection_methodology = str(provenance.get("collection_methodology", "")).strip()

    citations = coerce_list_of_text(meta.get("citations"))
    if not citations:
        citations = [default_citation(meta, dataset_ref)]

    return DatasetUiPayload(
        dataset_dir=dataset_dir,
        dataset_ref=dataset_ref,
        author_name=author_name,
        author_bio=author_bio,
        temporal_start_date=temporal_start_date,
        temporal_end_date=temporal_end_date,
        geospatial_coverage=geospatial_coverage,
        doi=doi,
        sources=sources,
        collection_methodology=collection_methodology,
        citations=citations,
    )


def discover_payloads(
    root: Path,
    *,
    dataset_dirs: set[str] | None,
    dataset_refs: set[str] | None,
    force_doi: str | None,
) -> list[DatasetUiPayload]:
    datasets_root = root / "datasets"
    if not datasets_root.exists():
        raise SystemExit(f"datasets/ directory not found under: {root}")

    payloads: list[DatasetUiPayload] = []
    for ds_dir in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
        dataset_name = ds_dir.name
        if dataset_dirs and dataset_name not in dataset_dirs:
            continue
        meta_path = ds_dir / "dataset-metadata.json"
        if not meta_path.exists():
            continue
        meta = load_json_object(meta_path)
        payload = build_payload(meta, dataset_name, force_doi=force_doi)
        if dataset_refs and payload.dataset_ref not in dataset_refs:
            continue
        payloads.append(payload)

    if not payloads:
        raise SystemExit("No dataset metadata payloads selected.")
    return payloads


def require_playwright():
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError as exc:
        raise SystemExit(
            "playwright is not installed. Run:\n"
            "  pip install -r pi-automation/scripts/requirements.txt\n"
            "  python -m playwright install chromium"
        ) from exc
    return sync_playwright, PlaywrightTimeout


def locator_count(locator) -> int:
    try:
        return locator.count()
    except Exception:
        return 0


def first_available(*locators):
    for locator in locators:
        if locator is not None and locator_count(locator):
            return locator
    return None


def is_login_prompt_visible(page) -> bool:
    login_markers = (
        page.get_by_role("link", name=re.compile(r"^sign in$", re.IGNORECASE)).first,
        page.get_by_role("button", name=re.compile(r"^sign in$", re.IGNORECASE)).first,
        page.get_by_role("link", name=re.compile(r"^register$", re.IGNORECASE)).first,
        page.locator('a[href*="/account/login"]').first,
    )
    return any(locator_count(marker) for marker in login_markers)


def is_authenticated(page) -> bool:
    url = str(getattr(page, "url", "") or "").lower()
    if "/account/login" in url:
        return False
    return not is_login_prompt_visible(page)


def section_visible(page, title: str) -> bool:
    heading = page.get_by_role("heading", name=re.compile(rf"^{re.escape(title)}$", re.IGNORECASE)).first
    if locator_count(heading):
        return True
    return locator_count(page.get_by_text(re.compile(rf"^{re.escape(title)}$", re.IGNORECASE)).first) > 0


def metadata_area_visible(page) -> bool:
    expand_all = page.get_by_role(
        "button",
        name=re.compile(r"(expand|collapse)\s+all\s+metadata\s+sections", re.IGNORECASE),
    ).first
    if locator_count(expand_all):
        return True
    metadata_heading = page.get_by_role("heading", name=re.compile(r"^metadata$", re.IGNORECASE)).first
    return locator_count(metadata_heading) > 0 and any(section_visible(page, title) for title in SECTION_TITLES)


def wait_for_metadata_area(page, timeout_ms: int) -> bool:
    deadline = time.time() + max(timeout_ms, 250) / 1000.0
    while time.time() < deadline:
        if metadata_area_visible(page):
            return True
        page.wait_for_timeout(250)
    return metadata_area_visible(page)


def find_editor_url(page, dataset_ref: str, timeout_ms: int) -> str:
    urls = [
        f"https://www.kaggle.com/datasets/{dataset_ref}/edit",
        f"https://www.kaggle.com/datasets/{dataset_ref}/settings",
        f"https://www.kaggle.com/datasets/{dataset_ref}/metadata",
        f"https://www.kaggle.com/datasets/{dataset_ref}",
    ]
    for url in urls:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(500)
        if wait_for_metadata_area(page, min(timeout_ms, 4000)) and is_authenticated(page):
            return page.url

    # Last attempt: dataset page then click an edit/settings affordance.
    page.goto(f"https://www.kaggle.com/datasets/{dataset_ref}", wait_until="domcontentloaded", timeout=timeout_ms)
    menu_button = first_available(
        page.get_by_role("button", name=re.compile(r"more options for this dataset", re.IGNORECASE)).first,
        page.get_by_role("button", name=re.compile(r"more options", re.IGNORECASE)).first,
    )
    if menu_button is not None:
        menu_button.click(timeout=timeout_ms)
        page.wait_for_timeout(350)

    edit_candidates = (
        page.get_by_role("menuitem", name=re.compile(r"edit|settings|metadata", re.IGNORECASE)).first,
        page.get_by_role("button", name=re.compile(r"edit|settings|metadata", re.IGNORECASE)).first,
        page.get_by_role("link", name=re.compile(r"edit|settings|metadata", re.IGNORECASE)).first,
    )
    for candidate in edit_candidates:
        if locator_count(candidate):
            candidate.click(timeout=timeout_ms)
            page.wait_for_timeout(800)
            if wait_for_metadata_area(page, min(timeout_ms, 4000)) and is_authenticated(page):
                return page.url

    if not is_authenticated(page):
        raise RuntimeError(
            "Kaggle session appears signed out. Re-run with --manual-login (headed) to refresh storage state."
        )
    raise RuntimeError(f"Could not find metadata editor for {dataset_ref}")


def maybe_login(page, *, email: str, password: str, manual_login: bool, timeout_ms: int) -> None:
    page.goto("https://www.kaggle.com/datasets", wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(500)
    if is_authenticated(page):
        return

    page.goto("https://www.kaggle.com/account/login", wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(500)
    if is_authenticated(page):
        return

    email_input = first_available(
        page.locator('input[name="email"]').first,
        page.locator('input[type="email"]').first,
    )
    password_input = first_available(
        page.locator('input[name="password"]').first,
        page.locator('input[type="password"]').first,
    )

    if email and password and email_input is not None and password_input is not None:
        email_input.fill(email, timeout=timeout_ms)
        password_input.fill(password, timeout=timeout_ms)
        submit_button = first_available(
            page.locator('button[type="submit"]').first,
            page.get_by_role("button", name=re.compile(r"sign in|log in", re.IGNORECASE)).first,
        )
        if submit_button is not None:
            submit_button.click(timeout=timeout_ms)
        else:
            page.keyboard.press("Enter")
        page.wait_for_timeout(1500)
        page.goto("https://www.kaggle.com/datasets", wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(500)
        if is_authenticated(page):
            return

    if manual_login:
        page.goto("https://www.kaggle.com/account/login", wait_until="domcontentloaded", timeout=timeout_ms)
        print("Manual login required: complete Kaggle login in the opened browser window.")
        input("Press Enter after login completes...")
        page.goto("https://www.kaggle.com/datasets", wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(500)
        if is_authenticated(page):
            return
        raise RuntimeError("Kaggle login still appears unauthenticated after manual login.")

    raise RuntimeError(
        "Kaggle login required but session appears signed out. "
        "Provide KAGGLE_EMAIL/KAGGLE_PASSWORD or run with --manual-login."
    )


def find_section_container(page, section_title: str):
    heading = page.get_by_role("heading", name=re.compile(rf"^{re.escape(section_title)}$", re.IGNORECASE)).first
    if locator_count(heading):
        return heading.locator("xpath=ancestor::*[self::section or self::article or self::div][1]").first
    text_match = page.get_by_text(re.compile(rf"^{re.escape(section_title)}$", re.IGNORECASE)).first
    if locator_count(text_match):
        return text_match.locator("xpath=ancestor::*[self::section or self::article or self::div][1]").first
    return None


def open_section_editor(page, section_title: str, timeout_ms: int):
    container = find_section_container(page, section_title)
    section_token = re.escape(section_title)
    named_action = re.compile(
        rf"(edit|expand|add).*\b{section_token}\b|\b{section_token}\b.*(edit|expand|add)",
        re.IGNORECASE,
    )
    direct_actions = (
        re.compile(rf"^edit\s+{section_token}$", re.IGNORECASE),
        re.compile(rf"^expand\s+{section_token}$", re.IGNORECASE),
        re.compile(rf"^add\s+{section_token}$", re.IGNORECASE),
    )
    candidates = []
    if container is not None:
        candidates.extend(
            [
                container.get_by_role("button", name=re.compile(r"^edit$", re.IGNORECASE)).first,
                container.get_by_role("button", name=re.compile(r"^expand$", re.IGNORECASE)).first,
                container.get_by_role("button", name=re.compile(r"^add$", re.IGNORECASE)).first,
                container.get_by_role("button", name=named_action).first,
            ]
        )
        for pattern in direct_actions:
            candidates.append(container.get_by_role("button", name=pattern).first)
    candidates.extend(
        [
            page.get_by_role("button", name=named_action).first,
            page.get_by_role("button", name=re.compile(r"^edit$", re.IGNORECASE)).first,
        ]
    )
    for pattern in direct_actions:
        candidates.append(page.get_by_role("button", name=pattern).first)
    for candidate in candidates:
        if locator_count(candidate):
            candidate.click(timeout=timeout_ms)
            page.wait_for_timeout(400)
            refreshed = find_section_container(page, section_title)
            return refreshed if refreshed is not None else container
    return container


def active_form_scope(page, section_container):
    dialog = page.get_by_role("dialog").last
    if locator_count(dialog):
        return dialog
    if section_container is not None:
        return section_container
    return page


def fill_field(scope, label_patterns: Iterable[str], value: str, timeout_ms: int) -> bool:
    if not value.strip():
        return False
    for pattern in label_patterns:
        locator = scope.get_by_label(re.compile(pattern, re.IGNORECASE)).first
        if locator_count(locator):
            locator.fill(value, timeout=timeout_ms)
            return True
    return False


def save_section(page, scope, apply: bool, timeout_ms: int) -> bool:
    if not apply:
        cancel = scope.get_by_role("button", name=re.compile(r"cancel|close", re.IGNORECASE)).first
        if locator_count(cancel):
            cancel.click(timeout=timeout_ms)
        else:
            page.keyboard.press("Escape")
        page.wait_for_timeout(250)
        return True

    candidates = (
        scope.get_by_role("button", name=re.compile(r"save|update|done", re.IGNORECASE)).first,
        page.get_by_role("button", name=re.compile(r"save|update|done", re.IGNORECASE)).first,
    )
    for candidate in candidates:
        if locator_count(candidate):
            candidate.click(timeout=timeout_ms)
            page.wait_for_timeout(600)
            return True
    return False


def sync_authors(page, payload: DatasetUiPayload, apply: bool, timeout_ms: int) -> SectionResult:
    container = open_section_editor(page, "Authors", timeout_ms)
    if container is None:
        return SectionResult(name="Authors", status="failed" if apply else "skipped", detail="section control not found")
    scope = active_form_scope(page, container)
    filled_name = fill_field(scope, ["Author Name", "Author"], payload.author_name, timeout_ms)
    filled_bio = fill_field(scope, ["Bio"], payload.author_bio, timeout_ms)
    if not (filled_name or filled_bio):
        return SectionResult(name="Authors", status="failed" if apply else "skipped", detail="fields not found")
    if not save_section(page, scope, apply, timeout_ms):
        return SectionResult(name="Authors", status="failed", detail="save action not found")
    return SectionResult(name="Authors", status="updated" if apply else "planned")


def sync_coverage(page, payload: DatasetUiPayload, apply: bool, timeout_ms: int) -> SectionResult:
    container = open_section_editor(page, "Coverage", timeout_ms)
    if container is None:
        return SectionResult(name="Coverage", status="failed" if apply else "skipped", detail="section control not found")
    scope = active_form_scope(page, container)
    filled = False
    filled |= fill_field(scope, ["Temporal Coverage Start Date", "Start Date"], payload.temporal_start_date, timeout_ms)
    filled |= fill_field(scope, ["Temporal Coverage End Date", "End Date"], payload.temporal_end_date, timeout_ms)
    filled |= fill_field(scope, ["Geospatial Coverage"], payload.geospatial_coverage, timeout_ms)
    if not filled:
        return SectionResult(name="Coverage", status="failed" if apply else "skipped", detail="fields not found")
    if not save_section(page, scope, apply, timeout_ms):
        return SectionResult(name="Coverage", status="failed", detail="save action not found")
    return SectionResult(name="Coverage", status="updated" if apply else "planned")


def sync_doi(page, payload: DatasetUiPayload, apply: bool, timeout_ms: int) -> SectionResult:
    if not payload.doi:
        return SectionResult(name="DOI Citation", status="skipped", detail="no DOI value in metadata")
    container = open_section_editor(page, "DOI Citation", timeout_ms)
    if container is None:
        return SectionResult(
            name="DOI Citation",
            status="failed" if apply else "skipped",
            detail="section control not found",
        )
    scope = active_form_scope(page, container)
    filled = fill_field(scope, ["DOI \\(Digital Object Identifier\\)", "DOI"], payload.doi, timeout_ms)
    if not filled:
        return SectionResult(name="DOI Citation", status="failed" if apply else "skipped", detail="DOI field not found")
    if not save_section(page, scope, apply, timeout_ms):
        return SectionResult(name="DOI Citation", status="failed", detail="save action not found")
    return SectionResult(name="DOI Citation", status="updated" if apply else "planned")


def sync_provenance(page, payload: DatasetUiPayload, apply: bool, timeout_ms: int) -> SectionResult:
    container = open_section_editor(page, "Provenance", timeout_ms)
    if container is None:
        return SectionResult(name="Provenance", status="failed" if apply else "skipped", detail="section control not found")
    scope = active_form_scope(page, container)
    sources_text = "\n".join(payload.sources)
    filled = False
    filled |= fill_field(scope, ["Sources"], sources_text, timeout_ms)
    filled |= fill_field(scope, ["Collection Methodology"], payload.collection_methodology, timeout_ms)
    if not filled:
        return SectionResult(name="Provenance", status="failed" if apply else "skipped", detail="fields not found")
    if not save_section(page, scope, apply, timeout_ms):
        return SectionResult(name="Provenance", status="failed", detail="save action not found")
    return SectionResult(name="Provenance", status="updated" if apply else "planned")


def sync_citations(page, payload: DatasetUiPayload, apply: bool, timeout_ms: int) -> SectionResult:
    citation_text = "\n".join(payload.citations).strip()
    if not citation_text:
        return SectionResult(name="Citations", status="skipped", detail="no citation values in metadata")
    container = open_section_editor(page, "Citations", timeout_ms)
    if container is None:
        return SectionResult(name="Citations", status="failed" if apply else "skipped", detail="section control not found")
    scope = active_form_scope(page, container)
    filled = fill_field(scope, ["Citations", "Citation"], citation_text, timeout_ms)
    if not filled:
        return SectionResult(name="Citations", status="failed" if apply else "skipped", detail="citation field not found")
    if not save_section(page, scope, apply, timeout_ms):
        return SectionResult(name="Citations", status="failed", detail="save action not found")
    return SectionResult(name="Citations", status="updated" if apply else "planned")


def sync_dataset(page, payload: DatasetUiPayload, *, apply: bool, timeout_ms: int) -> DatasetResult:
    editor_url = find_editor_url(page, payload.dataset_ref, timeout_ms=timeout_ms)
    sections = [
        sync_authors(page, payload, apply=apply, timeout_ms=timeout_ms),
        sync_coverage(page, payload, apply=apply, timeout_ms=timeout_ms),
        sync_doi(page, payload, apply=apply, timeout_ms=timeout_ms),
        sync_provenance(page, payload, apply=apply, timeout_ms=timeout_ms),
        sync_citations(page, payload, apply=apply, timeout_ms=timeout_ms),
    ]
    return DatasetResult(dataset_ref=payload.dataset_ref, editor_url=editor_url, sections=sections)


def parse_comma_set(values: list[str]) -> set[str]:
    parsed: set[str] = set()
    for value in values:
        for item in value.split(","):
            text = item.strip().lower()
            if text:
                parsed.add(text)
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync Kaggle dataset UI metadata sections (authors/coverage/doi/provenance/citations)."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset directory name(s), comma-separated allowed.")
    parser.add_argument("--dataset-ref", action="append", default=[], help="Kaggle dataset ref owner/slug, comma-separated allowed.")
    parser.add_argument("--plan-only", action="store_true", help="Print selected metadata payloads and exit without browser automation.")
    parser.add_argument("--apply", action="store_true", help="Persist changes in Kaggle UI.")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode.")
    parser.add_argument("--slow-mo-ms", type=int, default=0, help="Delay each browser action by N ms.")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="Playwright action timeout in ms.")
    parser.add_argument("--storage-state", type=Path, default=DEFAULT_STORAGE_STATE, help="Playwright storage state JSON.")
    parser.add_argument("--email", default=os.environ.get("KAGGLE_EMAIL", ""), help="Kaggle login email.")
    parser.add_argument("--password", default=os.environ.get("KAGGLE_PASSWORD", ""), help="Kaggle login password.")
    parser.add_argument(
        "--manual-login",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow interactive login if credentials are not provided.",
    )
    parser.add_argument("--force-doi", default=None, help="Override DOI value for all selected datasets.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional output file for run report JSON.")
    return parser.parse_args(argv)


def storage_state_has_kaggle_cookie(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json_object(path)
    except Exception:
        return False
    cookies = payload.get("cookies")
    if not isinstance(cookies, list):
        return False
    for item in cookies:
        if isinstance(item, dict) and "kaggle.com" in str(item.get("domain", "")).lower():
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dirs = parse_comma_set(args.dataset)
    dataset_refs = parse_comma_set(args.dataset_ref)

    payloads = discover_payloads(
        args.root.resolve(),
        dataset_dirs=dataset_dirs or None,
        dataset_refs=dataset_refs or None,
        force_doi=args.force_doi,
    )

    print(f"Selected {len(payloads)} dataset(s):")
    for payload in payloads:
        print(f"  - {payload.dataset_ref} ({payload.dataset_dir})")

    if args.plan_only:
        plan = [asdict(payload) for payload in payloads]
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps({"plan_only": True, "datasets": plan}, indent=2), encoding="utf-8")
            print(f"\nPlan written: {args.report_json}")
        return 0

    state_path = args.storage_state.resolve()
    if (
        not args.email.strip()
        and not args.password.strip()
        and not args.manual_login
        and not storage_state_has_kaggle_cookie(state_path)
    ):
        raise SystemExit(
            "Kaggle login required: storage state has no Kaggle auth cookies and --no-manual-login was used. "
            "Run once with --headed --manual-login to refresh login state."
        )

    sync_playwright, PlaywrightTimeout = require_playwright()
    results: list[DatasetResult] = []
    with sync_playwright() as p:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_arg = str(state_path) if state_path.exists() else None
        browser = p.chromium.launch(headless=not args.headed, slow_mo=args.slow_mo_ms)
        context = browser.new_context(storage_state=state_arg)
        page = context.new_page()

        try:
            maybe_login(
                page,
                email=args.email.strip(),
                password=args.password.strip(),
                manual_login=args.manual_login,
                timeout_ms=args.timeout_ms,
            )
            context.storage_state(path=str(state_path))
        except Exception as exc:
            browser.close()
            raise SystemExit(f"Authentication failed: {exc}") from exc

        for payload in payloads:
            print(f"\nSyncing {payload.dataset_ref} ...")
            try:
                result = sync_dataset(page, payload, apply=args.apply, timeout_ms=args.timeout_ms)
                results.append(result)
                for section in result.sections:
                    detail = f" ({section.detail})" if section.detail else ""
                    print(f"  [{section.status}] {section.name}{detail}")
            except PlaywrightTimeout as exc:
                print(f"  [failed] timeout: {exc}")
                results.append(
                    DatasetResult(
                        dataset_ref=payload.dataset_ref,
                        editor_url=page.url,
                        sections=[SectionResult(name="run", status="failed", detail=f"timeout: {exc}")],
                    )
                )
            except Exception as exc:
                print(f"  [failed] {exc}")
                results.append(
                    DatasetResult(
                        dataset_ref=payload.dataset_ref,
                        editor_url=page.url,
                        sections=[SectionResult(name="run", status="failed", detail=str(exc))],
                    )
                )

        browser.close()

    report = {
        "apply": args.apply,
        "datasets": [
            {
                "dataset_ref": item.dataset_ref,
                "editor_url": item.editor_url,
                "sections": [asdict(section) for section in item.sections],
            }
            for item in results
        ],
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written: {args.report_json}")

    failed_count = sum(
        1
        for dataset in results
        for section in dataset.sections
        if section.status == "failed"
    )
    if failed_count:
        print(f"\nCompleted with {failed_count} failed section update(s).")
        return 1

    print("\nCompleted successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
