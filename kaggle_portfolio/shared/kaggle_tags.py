"""Kaggle's tag vocabulary — the only keywords a push will actually apply.

Kaggle validates `keywords` against a fixed list of 833 tags. Anything else is
**silently dropped**: the CLI prints "The following are not valid tags and could
not be added to the kernel" and the upload otherwise succeeds, so a repo can
carry keywords that have never once reached the site.

That is not hypothetical here. 34% of this repo's declared notebook keywords are
outside the vocabulary, and an earlier pass at "keeping the most searchable
terms" deleted 31 valid tags while keeping 64 invented ones — `education`, `eda`
and `deep learning` are real tags; `lora`, `shap` and `polars` are not.

Two further facts worth knowing before editing keywords:

* **Tags are additive.** A push never removes a tag a notebook already carries,
  so deleting a keyword here cannot undo a past push — it can only fail to add.
* **Titles are what search reads.** This account ranks #1 for "duckdb",
  "polars kaggle" and "rag from scratch", none of which are valid tags; they are
  in the titles. Put specific terms there, and spend tags on real vocabulary.
"""

from __future__ import annotations

import functools
from pathlib import Path

VOCABULARY_PATH = Path(__file__).with_name("kaggle_tags.tsv")

# Two different limits, both real, easy to conflate:
#   * UPLOAD accepts 6. Measured against the live API on 2026-08-19 — 7 is
#     rejected with 'You have exceeded the max category limit'. That is
#     `manage_commands.MAX_KEYWORDS`, and exceeding it breaks the push.
#   * KAGGLE APPLIES 5. Inferred, not measured: all 11 datasets declare 6 and
#     no live dataset or notebook carries more than 5 (58/58 observations).
# So a 6th keyword uploads cleanly and simply never appears. It is waste, not
# a failure, which is why nothing errors on it.
MAX_APPLIED_KEYWORDS = 5


@functools.lru_cache(maxsize=1)
def _vocabulary() -> dict[str, int]:
    """Lower-cased tag name and slug -> kernel count, for popularity ordering."""
    table: dict[str, int] = {}
    for line in VOCABULARY_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or line.startswith("name\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, slug, kernels = parts[0], parts[1], parts[2]
        try:
            count = int(kernels)
        except ValueError:
            count = 0
        table[name.strip().lower()] = count
        table[slug.strip().lower()] = count
    return table


def is_valid_tag(keyword: str) -> bool:
    """True when Kaggle will actually apply this keyword."""
    return str(keyword).strip().lower() in _vocabulary()


def invalid_tags(keywords) -> list[str]:
    """The keywords Kaggle would silently drop, in the order given."""
    return [k for k in (keywords or []) if not is_valid_tag(k)]


def suggest(term: str, limit: int = 5) -> list[str]:
    """Valid tags containing `term`, most-used first — for replacing a rejected one."""
    needle = str(term).strip().lower()
    hits = [(name, n) for name, n in _vocabulary().items() if needle in name]
    hits.sort(key=lambda pair: (-pair[1], pair[0]))
    seen, out = set(), []
    for name, _ in hits:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= limit:
            break
    return out
