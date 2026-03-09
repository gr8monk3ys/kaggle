from datetime import date

import deadline_alert

SAMPLE_TRACKER = """
## Current Progress (2026-01-25)

### Competitions
| Status | Target | Current |
|--------|--------|---------|
| Tier | Grandmaster | Novice |

**Active competitions to enter:**
| Competition | Teams | Deadline | Medal Difficulty | Strategy |
|-------------|-------|----------|-----------------|----------|
| Med-Gemma Impact Challenge | 58 | Feb 24, 2026 | Easiest | Fine-tune |
| Akkadian Translation | 1321 | Mar 23, 2026 | Hard | ByT5 |
| Already Expired | 100 | Jan 01, 2026 | Easy | skip |
"""


def test_parse_deadlines_finds_all_rows():
    today = date(2026, 2, 22)
    deadlines = deadline_alert.parse_deadlines(SAMPLE_TRACKER, today)
    assert len(deadlines) == 3


def test_filter_urgent_finds_within_72h():
    today = date(2026, 2, 22)
    deadlines = deadline_alert.parse_deadlines(SAMPLE_TRACKER, today)
    urgent = deadline_alert.filter_urgent(deadlines, hours=72)
    assert len(urgent) == 1
    assert urgent[0].competition == "Med-Gemma Impact Challenge"


def test_filter_urgent_excludes_past():
    today = date(2026, 2, 22)
    deadlines = deadline_alert.parse_deadlines(SAMPLE_TRACKER, today)
    urgent = deadline_alert.filter_urgent(deadlines, hours=72)
    names = [d.competition for d in urgent]
    assert "Already Expired" not in names


def test_format_alert_contains_key_fields():
    today = date(2026, 2, 22)
    deadlines = deadline_alert.parse_deadlines(SAMPLE_TRACKER, today)
    urgent = deadline_alert.filter_urgent(deadlines, hours=72)
    msg = deadline_alert.format_alert(urgent[0])
    assert "Med-Gemma" in msg
    assert "Feb 24" in msg
