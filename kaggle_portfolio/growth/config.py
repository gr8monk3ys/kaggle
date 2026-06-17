"""Configuration leaf for the growth flywheel. No intra-package imports."""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

# Medal vote cuts: bronze, silver, gold.
CUTS: tuple[int, int, int] = (5, 20, 50)
# Relative value of reaching each cut (gold worth most).
TIER_VALUE: dict[int, float] = {5: 1.0, 20: 2.5, 50: 6.0}


@dataclass(frozen=True)
class FlywheelConfig:
    w_followers: float = 5.0
    w_votes: float = 1.0
    w_discussion: float = 0.5
    near_bonus: float = 3.0
    near_window: int = 3
    max_posts_per_day: int = 2
    max_posts_per_week: int = 8
    max_forum_drops_per_comp_per_week: int = 1
    window_start_hour: int = 13  # inclusive UTC hour
    window_end_hour: int = 23    # inclusive UTC hour
    enabled: bool = True
    ema_alpha: float = 0.3


def load_config(path: Path) -> FlywheelConfig:
    """Load config from JSON, ignoring unknown keys; missing file -> defaults."""
    if not path.exists():
        return FlywheelConfig()
    raw = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(FlywheelConfig)}
    return FlywheelConfig(**{k: v for k, v in raw.items() if k in known})
