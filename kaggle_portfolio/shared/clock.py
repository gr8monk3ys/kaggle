#!/usr/bin/env python3
"""The clock, as an injectable seam.

Every ``datetime.now()`` and ``date.today()`` in a command path should come from
here, so that ``--today`` works uniformly and tests never depend on the wall
clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone


def parse_iso_date(text: str | None) -> date | None:
    """Parse a ``YYYY-MM-DD`` string into a :class:`date`, or ``None``."""
    if not text:
        return None
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


class InvalidDateOverride(ValueError):
    """Raised when a ``--today`` override is not a valid ``YYYY-MM-DD`` date."""


@dataclass(frozen=True)
class Clock:
    """Today's date and the current instant, fixed for the life of a command.

    ``today`` is resolved once at construction so that a command spanning
    midnight cannot see two different dates.
    """

    today: date
    _now: datetime | None = None

    @classmethod
    def resolve(cls, today_override: str | None = None) -> "Clock":
        """Build a clock, honouring a ``YYYY-MM-DD`` override when given."""
        if not today_override:
            return cls(today=date.today())
        parsed = parse_iso_date(today_override)
        if parsed is None:
            raise InvalidDateOverride(f"Invalid --today value: {today_override}")
        return cls(
            today=parsed,
            _now=datetime.combine(parsed, datetime.min.time(), tzinfo=timezone.utc),
        )

    @classmethod
    def fixed(cls, day: date | str) -> "Clock":
        """Build a clock pinned to *day*. For tests."""
        return cls.resolve(day if isinstance(day, str) else day.isoformat())

    def now(self) -> datetime:
        """Return the current UTC instant, or the pinned instant when overridden."""
        if self._now is not None:
            return self._now
        return datetime.now(tz=timezone.utc)

    def isoformat(self) -> str:
        return self.today.isoformat()
