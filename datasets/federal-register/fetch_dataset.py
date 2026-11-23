#!/usr/bin/env python3
"""Snapshot every document published in the US Federal Register over a date window.

Real data, not generated: every row comes from the Federal Register's public
API (federalregister.gov/developers), which needs no key. The documents
themselves do not change once published, but ``page_views`` and
``significant`` do, so the CSV this writes is committed as source and the
script is named ``fetch_`` rather than ``create_`` to keep it out of
``./manage.sh build-datasets``.

Usage:
    python fetch_dataset.py                                   # default window (~10 minutes)
    python fetch_dataset.py --start 2026-10-01 --end 2026-10-31   # quick smoke run
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "federal_register_documents.csv"
API = "https://www.federalregister.gov/api/v1/documents.json"
PAUSE_SECONDS = 0.5

API_FIELDS = (
    "document_number",
    "publication_date",
    "type",
    "subtype",
    "title",
    "abstract",
    "action",
    "agencies",
    "significant",
    "page_length",
    "effective_on",
    "comments_close_on",
    "topics",
    "cfr_references",
    "docket_ids",
    "regulation_id_numbers",
    "president",
    "executive_order_number",
    "signing_date",
    "correction_of",
    "page_views",
)

FIELDS = (
    "document_number",
    "publication_date",
    "type",
    "subtype",
    "title",
    "abstract",
    "action",
    "agency",
    "agencies",
    "n_agencies",
    "significant",
    "page_length",
    "effective_on",
    "days_to_effective",
    "comments_close_on",
    "comment_period_days",
    "topics",
    "n_topics",
    "cfr_titles",
    "n_cfr_references",
    "n_dockets",
    "regulation_id_numbers",
    "president",
    "executive_order_number",
    "signing_date",
    "is_correction",
    "page_views",
)


def get(url: str, attempts: int = 4) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": "dataset-snapshot/1.0"}
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        # Pages are up to ~1 MB of JSON; a dropped connection mid-body is transient.
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows = []
    cursor = start
    while cursor <= end:
        nxt = date(cursor.year + cursor.month // 12, cursor.month % 12 + 1, 1)
        windows.append((cursor, min(end, date.fromordinal(nxt.toordinal() - 1))))
        cursor = nxt
    return windows


def fetch_window(first: date, last: date) -> list[dict]:
    """All documents published in [first, last], following the API's cursor."""
    params = [
        ("per_page", "1000"),
        ("order", "oldest"),
        ("conditions[publication_date][gte]", first.isoformat()),
        ("conditions[publication_date][lte]", last.isoformat()),
        *(("fields[]", field) for field in API_FIELDS),
    ]
    url: str | None = f"{API}?{urllib.parse.urlencode(params)}"
    docs: list[dict] = []
    expected = None
    while url:
        page = get(url)
        expected = page.get("count", 0)
        docs.extend(page.get("results") or [])
        url = page.get("next_page_url")
        time.sleep(PAUSE_SECONDS)
    if len(docs) != expected:
        raise RuntimeError(
            f"{first}..{last}: API reported {expected} documents, got {len(docs)}"
        )
    return docs


def days(later: str | None, earlier: str) -> int | None:
    if not later:
        return None
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


def unique(values: list) -> list:
    return list(dict.fromkeys(v for v in values if v not in (None, "")))


def clean(text: str | None) -> str:
    return " ".join((text or "").split())


def to_row(doc: dict) -> dict:
    agencies = doc.get("agencies") or []
    names = unique([a.get("name") or a.get("raw_name") for a in agencies])
    # The parent department when one is listed, else the first agency named.
    top = next(
        (
            a.get("name")
            for a in agencies
            if a.get("parent_id") is None and a.get("name")
        ),
        None,
    )
    cfr = doc.get("cfr_references") or []
    topics = unique(doc.get("topics") or [])
    published = doc["publication_date"]
    return {
        "document_number": doc["document_number"],
        "publication_date": published,
        "type": doc.get("type") or "",
        "subtype": doc.get("subtype") or "",
        "title": clean(doc.get("title")),
        "abstract": clean(doc.get("abstract")),
        "action": clean(doc.get("action")),
        "agency": top or (names[0] if names else ""),
        "agencies": "|".join(names),
        "n_agencies": len(names),
        "significant": "" if doc.get("significant") is None else doc["significant"],
        "page_length": doc.get("page_length"),
        "effective_on": doc.get("effective_on") or "",
        "days_to_effective": days(doc.get("effective_on"), published),
        "comments_close_on": doc.get("comments_close_on") or "",
        "comment_period_days": days(doc.get("comments_close_on"), published),
        "topics": "|".join(topics),
        "n_topics": len(topics),
        "cfr_titles": "|".join(str(t) for t in unique([c.get("title") for c in cfr])),
        "n_cfr_references": len(cfr),
        "n_dockets": len(doc.get("docket_ids") or []),
        "regulation_id_numbers": "|".join(
            unique(doc.get("regulation_id_numbers") or [])
        ),
        "president": (doc.get("president") or {}).get("name") or "",
        "executive_order_number": doc.get("executive_order_number") or "",
        "signing_date": doc.get("signing_date") or "",
        "is_correction": bool(doc.get("correction_of")),
        "page_views": (doc.get("page_views") or {}).get("count"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 31))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    docs: dict[str, dict] = {}
    for first, last in month_windows(args.start, args.end):
        found = fetch_window(first, last)
        docs.update((d["document_number"], d) for d in found)
        print(
            f"{first:%Y-%m}  {len(found):>5} documents  ({len(docs):>6} total)",
            flush=True,
        )

    rows = sorted(
        (to_row(d) for d in docs.values()),
        key=lambda r: (r["publication_date"], r["document_number"]),
    )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Wrote {len(rows)} documents to {args.output.name} (snapshot {datetime.now(timezone.utc):%Y-%m-%d})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
