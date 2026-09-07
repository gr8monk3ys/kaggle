#!/usr/bin/env python3
"""Where the repo is, and what lives where in it.

Seventeen modules used to derive the repo root independently with
``Path(__file__).resolve().parents[2]``, and only one of them honoured the
``KAGGLE_DIR`` override. This module is the single answer, constructed once at
the CLI edge and passed down.

Every path here is anchored to the repo root, never to the current working
directory. Command modules previously defaulted to relative paths such as
``Path("medal_ops")``, which resolved correctly only because subprocess
delegation forced the working directory; anchoring them removes that dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

METADATA_NAMES = {"kernel-metadata.json", "dataset-metadata.json"}


@dataclass
class RepoLayout:
    """The repo root plus every path derived from it.

    Directory discovery is lazy and cached per instance, not per process: a
    long-lived interpreter running several commands gets a fresh scan for each,
    so creating a notebook and then pushing it in the same session works.
    """

    root: Path
    output_root: Path
    _notebook_dirs: list[str] | None = field(default=None, repr=False, compare=False)
    _dataset_dirs: list[str] | None = field(default=None, repr=False, compare=False)

    # -- construction -------------------------------------------------------

    @classmethod
    def resolve(
        cls, root: Path | str | None = None, output_root: Path | str | None = None
    ) -> "RepoLayout":
        """Build a layout, honouring ``KAGGLE_DIR`` when no explicit root is given."""
        if root is None:
            env_root = os.environ.get("KAGGLE_DIR", "").strip()
            base = Path(env_root) if env_root else Path(__file__).resolve().parents[2]
        else:
            base = Path(root)
        base = base.resolve()
        out = Path(output_root) if output_root is not None else base / "medal_ops"
        if not out.is_absolute():
            out = base / out
        return cls(root=base, output_root=out)

    def with_output_root(self, output_root: Path | str) -> "RepoLayout":
        """Return a copy writing reports elsewhere. Used by preflight."""
        return RepoLayout.resolve(self.root, output_root)

    # -- named paths --------------------------------------------------------

    @property
    def tracker_path(self) -> Path:
        return self.root / "docs" / "reports" / "grandmaster-tracker.md"

    @property
    def reports_dir(self) -> Path:
        return self.output_root / "reports"

    @property
    def leaderboard_dir(self) -> Path:
        return self.output_root / "leaderboard"

    @property
    def scout_report(self) -> Path:
        return self.root / "docs" / "reports" / "competition-scout-report.md"

    @property
    def drafts_path(self) -> Path:
        return self.root / "docs" / "discussions" / "discussion-drafts.md"

    @property
    def pi_root(self) -> Path:
        return self.root / "pi-automation"

    @property
    def pi_scripts(self) -> Path:
        return self.pi_root / "scripts"

    @property
    def pi_data(self) -> Path:
        return self.pi_root / "data"

    @property
    def queue_path(self) -> Path:
        return self.pi_data / "discussion_queue.json"

    @property
    def campaign_queue_path(self) -> Path:
        return self.pi_data / "promotion_campaign_queue.json"

    @property
    def storage_state_path(self) -> Path:
        return self.pi_data / "kaggle_storage_state.json"

    @property
    def growth_dir(self) -> Path:
        env_dir = os.environ.get("FLYWHEEL_DIR", "").strip()
        return Path(env_dir) if env_dir else self.root / "medal_ops" / "growth"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def competitions_dir(self) -> Path:
        return self.root / "projects" / "competitions"

    @property
    def lab_root(self) -> Path:
        return self.root / ".competition_lab"

    @property
    def local_credentials(self) -> Path:
        return self.root / "kaggle.json"

    @property
    def manage_script(self) -> Path:
        return self.root / "manage.sh"

    # -- discovery ----------------------------------------------------------

    def is_skipped(self, path: Path) -> bool:
        """True when *path* lies inside a skipped directory, judged relative to the root.

        Relative, not absolute: the checkout itself can sit under a hidden
        directory (agent worktrees live in ``.claude/worktrees/``), and matching
        on absolute parts would then skip every file in the repo.
        """
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return True
        # Any dot-directory, plus __pycache__. A superset of the two rules this
        # replaces: an explicit skip list here, and "any part starts with a dot"
        # in metadata_tracker. Scratch dirs like .competition_lab are covered
        # without having to be enumerated.
        return any(part.startswith(".") or part == "__pycache__" for part in rel.parts)

    def kernel_metadata_dirs(
        self, *, include_dataset_notebooks: bool = False
    ) -> list[str]:
        """Repo-relative directories holding a ``kernel-metadata.json``.

        Eleven ``datasets/*`` folders carry one too, for their explore notebooks.
        They are excluded by default because *pushing* those directories is a
        dataset operation, not a kernel one — but anything measuring notebooks
        (votes, metadata drift) wants them, hence the flag.
        """
        items: list[str] = []
        for meta in sorted(self.root.rglob("kernel-metadata.json")):
            if self.is_skipped(meta):
                continue
            try:
                rel = meta.parent.relative_to(self.root)
            except ValueError:
                continue
            if (
                not include_dataset_notebooks
                and rel.parts
                and rel.parts[0] == "datasets"
            ):
                continue
            items.append(str(rel))
        return items

    def notebook_dirs(self) -> list[str]:
        """Repo-relative directories a ``push`` treats as notebooks."""
        if self._notebook_dirs is None:
            self._notebook_dirs = self.kernel_metadata_dirs()
        return list(self._notebook_dirs)

    def dataset_dirs(self) -> list[str]:
        """Repo-relative directories holding a ``dataset-metadata.json``."""
        if self._dataset_dirs is None:
            items: list[str] = []
            for meta in sorted(self.root.rglob("dataset-metadata.json")):
                if self.is_skipped(meta):
                    continue
                try:
                    items.append(str(meta.parent.relative_to(self.root)))
                except ValueError:
                    continue
            self._dataset_dirs = items
        return list(self._dataset_dirs)

    def rescan(self) -> None:
        """Drop cached discovery so the next call re-walks the repo."""
        self._notebook_dirs = None
        self._dataset_dirs = None
