#!/usr/bin/env python3
"""The Kaggle seam: one interface, two adapters.

Callers ask for Kaggle nouns and get typed results. Everything about *how* we
talk to Kaggle — subprocess invocation, the CSV noise banner, pagination and its
version quirks, retry, credential resolution, and the SDK path used for the
upload-auth probe — lives behind this interface.

See ``docs/adr/0001-kaggle-cli-behind-one-adapter.md``.

Two adapters make the seam real: :class:`CliKaggleClient` in production and
:class:`FakeKaggleClient` everywhere else. The fake ships in the package rather
than in ``tests/`` so that ``pi-automation``'s suite and the dry-run paths can
use it too.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class KaggleError(RuntimeError):
    """Anything that went wrong talking to Kaggle.

    One exception type, deliberately. Failure used to mean five different things
    across fifteen call sites, and roughly eight of them swallowed it. A caller
    that genuinely wants to tolerate a failure now writes the ``except``.
    """


class KaggleCliMissing(KaggleError):
    """No usable Kaggle CLI is available."""


class KaggleCommandFailed(KaggleError):
    """A Kaggle CLI command exited non-zero."""

    def __init__(self, argv: Sequence[str], stdout: str, stderr: str) -> None:
        self.argv = list(argv)
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"{' '.join(self.argv)}: {summarize_output(stdout, stderr)}")


class KaggleFieldMissing(KaggleError):
    """Kaggle returned rows without a column we require.

    This is the loud version of the bug that used to be silent: a ``Warning:``
    banner parsed as a header row yields dicts with garbage keys, and every
    downstream lookup returns ``None`` without anyone noticing.
    """


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

#: Line prefixes Kaggle writes to stdout that are not part of the CSV payload.
NOISE_PREFIXES = ("Warning:", "Next Page Token", "/")


def summarize_output(*chunks: str) -> str:
    """Return a compact, human-meaningful summary of command output."""
    import re

    lines = [
        line.strip() for chunk in chunks for line in chunk.splitlines() if line.strip()
    ]
    if not lines:
        return "unknown error"
    preferred = [
        line
        for line in lines
        if re.search(
            r"(error|unauthorized|forbidden|denied|failed|exception|traceback)",
            line,
            re.IGNORECASE,
        )
    ]
    return (preferred[-1] if preferred else lines[-1])[:220]


def strip_noise(text: str) -> str:
    """Drop Kaggle's banner lines so the first surviving line is the CSV header.

    Four call sites used to do this four different ways, and eight did not do it
    at all.
    """
    kept = [
        line
        for line in text.splitlines()
        if line.strip() and not line.startswith(NOISE_PREFIXES)
    ]
    return "\n".join(kept)


def parse_csv(text: str) -> list[dict[str, str]]:
    """Parse Kaggle CSV output into rows, after stripping banner noise."""
    cleaned = strip_noise(text)
    if not cleaned.strip():
        return []
    return [dict(row) for row in csv.DictReader(io.StringIO(cleaned))]


def _normalise(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def pick(
    row: dict[str, str], candidates: Sequence[str], *, what: str, required: bool = True
) -> Any:
    """Return the first candidate column present in *row*.

    Kaggle's column spellings drift across CLI versions, which is why the
    fallback list exists at all. It exists exactly once, here, and its failure
    is loud rather than a ``None`` that propagates.
    """
    normalised = {_normalise(k): v for k, v in row.items()}
    for candidate in candidates:
        key = _normalise(candidate)
        if key in normalised:
            return normalised[key]
    for candidate in candidates:
        key = _normalise(candidate)
        for actual, value in normalised.items():
            if key in actual:
                return value
    if required:
        raise KaggleFieldMissing(
            f"No column for {what} in Kaggle output. Tried {list(candidates)}; got {sorted(row)}"
        )
    return None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Kernel:
    ref: str
    title: str
    total_votes: int
    raw: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def slug(self) -> str:
        return self.ref.rsplit("/", 1)[-1]

    @property
    def is_private_placeholder(self) -> bool:
        """Kaggle lists private notebooks with a literal placeholder title."""
        return self.title.strip().lower() == "[private notebook]"

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Kernel":
        return cls(
            ref=str(pick(row, ("ref",), what="kernel ref")).strip(),
            title=str(
                pick(row, ("title",), what="kernel title", required=False) or ""
            ).strip(),
            total_votes=_as_int(
                pick(
                    row,
                    ("totalVotes", "voteCount", "votes", "vote"),
                    what="kernel votes",
                )
            ),
            raw=row,
        )


@dataclass(frozen=True)
class Dataset:
    ref: str
    title: str
    size: str
    last_updated: str
    download_count: int
    vote_count: int
    usability_rating: float | None
    raw: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def slug(self) -> str:
        return self.ref.rsplit("/", 1)[-1]

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Dataset":
        return cls(
            ref=str(pick(row, ("ref",), what="dataset ref")).strip(),
            title=str(
                pick(row, ("title",), what="dataset title", required=False) or ""
            ).strip(),
            size=str(
                pick(row, ("size",), what="dataset size", required=False) or ""
            ).strip(),
            last_updated=str(
                pick(row, ("lastUpdated",), what="dataset lastUpdated", required=False)
                or ""
            ).strip(),
            download_count=_as_int(
                pick(
                    row,
                    ("downloadCount", "downloads", "download"),
                    what="dataset downloads",
                    required=False,
                )
            ),
            vote_count=_as_int(
                pick(
                    row,
                    ("totalVotes", "voteCount", "votes", "vote"),
                    what="dataset votes",
                    required=False,
                )
            ),
            usability_rating=_as_float(
                pick(
                    row,
                    ("usabilityRating", "usability"),
                    what="dataset usability",
                    required=False,
                )
            ),
            raw=row,
        )


@dataclass(frozen=True)
class Competition:
    ref: str
    title: str
    category: str
    team_count: int
    deadline: str
    user_has_entered: bool
    raw: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def slug(self) -> str:
        return self.ref.rsplit("/", 1)[-1]

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Competition":
        entered = pick(
            row,
            ("userHasEntered", "hasEntered", "entered"),
            what="competition entered flag",
            required=False,
        )
        return cls(
            ref=str(pick(row, ("ref",), what="competition ref")).strip(),
            title=str(
                pick(row, ("title",), what="competition title", required=False) or ""
            ).strip(),
            category=str(
                pick(row, ("category",), what="competition category", required=False)
                or ""
            ).strip(),
            team_count=_as_int(
                pick(
                    row,
                    ("teamCount", "team_count"),
                    what="competition teams",
                    required=False,
                )
            ),
            deadline=str(
                pick(
                    row,
                    ("deadline", "evaluationDate"),
                    what="competition deadline",
                    required=False,
                )
                or ""
            ).strip(),
            user_has_entered=str(entered).strip().lower() in {"true", "1", "yes"},
            raw=row,
        )


@dataclass(frozen=True)
class LeaderboardEntry:
    rank: int
    team_name: str
    score: float | None


@dataclass(frozen=True)
class Submission:
    public_score: float | None
    raw: dict[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class DatasetMetadata:
    title: str
    subtitle: str
    description: str
    usability_rating: float | None
    total_votes: int
    total_downloads: int
    is_private: bool | None
    keywords: list[str]
    licenses: list[dict[str, Any]]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class PushOutcome:
    """The result of a mutating operation."""

    ok: bool
    detail: str = ""
    skipped: bool = False  # True when effects were disabled

    def __bool__(self) -> bool:
        return self.ok


@dataclass(frozen=True)
class Credentials:
    username: str | None
    key: str | None
    source: str

    def masked(self) -> str:
        if not self.key:
            return "(none)"
        return (
            self.key[:4] + "…" + self.key[-4:]
            if len(self.key) > 8
            else "*" * len(self.key)
        )


@dataclass(frozen=True)
class CredentialState:
    credentials: Credentials | None
    sources: list[str]
    error: str | None = None

    def __bool__(self) -> bool:
        return self.credentials is not None


@dataclass(frozen=True)
class AuthProbe:
    ok: bool
    detail: str


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------


class KaggleClient(Protocol):
    """What callers may know about Kaggle.

    Adding a method here is a deliberate widening of the seam. Prefer expressing
    a new need in terms of an existing noun.
    """

    # discovery / auth
    def available(self) -> bool: ...
    def credentials(self) -> CredentialState: ...
    def probe_upload_auth(self, timeout: int = ...) -> AuthProbe: ...

    # kernels
    def my_kernels(self, *, kernel_type: str | None = ...) -> list[Kernel]: ...
    def push_kernel(self, path: Path) -> PushOutcome: ...

    # datasets
    def my_datasets(self) -> list[Dataset]: ...
    def datasets_by_owner(self, owner: str) -> list[Dataset]: ...
    def search_datasets(
        self, *, sort_by: str | None = ..., pages: int = ...
    ) -> list[Dataset]: ...
    def dataset_metadata(self, ref: str, dest: Path) -> DatasetMetadata: ...
    def dataset_files(self, ref: str) -> list[str]: ...
    def publish_dataset(self, path: Path, message: str) -> PushOutcome: ...

    # competitions
    def entered_competitions(self) -> list[Competition]: ...
    def search_competitions(
        self, *, category: str | None = ..., sort_by: str = ...
    ) -> list[Competition]: ...
    def leaderboard(self, slug: str) -> list[LeaderboardEntry]: ...
    def submissions(self, slug: str) -> list[Submission]: ...
    def download_competition(self, slug: str, dest: Path) -> Path: ...
    def submit(self, slug: str, file: Path, message: str) -> PushOutcome: ...


# ---------------------------------------------------------------------------
# Production adapter
# ---------------------------------------------------------------------------

#: Kaggle's default page size when ``--page-size`` is unavailable.
FALLBACK_PAGE_SIZE = 20
#: What we ask for when it is available.
PREFERRED_PAGE_SIZE = 100


class CliKaggleClient:
    """Talks to Kaggle through its CLI, with the SDK used only for the auth probe.

    ``effects=False`` turns every mutating method into a recorded no-op, so
    ``--dry-run`` is enforced here rather than by a conditional each caller has
    to remember to write.
    """

    def __init__(
        self, *, effects: bool = True, retries: int = 3, timeout: int | None = None
    ) -> None:
        self._effects = effects
        self._retries = max(1, retries)
        self._timeout = timeout
        self._page_size_ok: bool | None = None
        self.skipped_effects: list[str] = []

    # -- plumbing -----------------------------------------------------------

    @staticmethod
    def _cli_module_available() -> bool:
        import importlib.util

        try:
            return importlib.util.find_spec("kaggle.cli") is not None
        except (ImportError, OSError):
            return False

    @staticmethod
    def _cli_path() -> str | None:
        candidates: list[Path] = []
        override = os.environ.get("KAGGLE_CLI_BIN", "").strip()
        if override:
            candidates.append(Path(override).expanduser())
        which = shutil.which("kaggle")
        if which:
            candidates.append(Path(which))
        exe = Path(sys.executable)
        candidates += [
            exe.parent / "kaggle",
            exe.parent / "kaggle.exe",
            exe.resolve().parent / "kaggle",
            exe.resolve().parent / "kaggle.exe",
        ]
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return None

    def _prefix(self) -> list[str]:
        binary = self._cli_path()
        if binary:
            return [binary]
        if self._cli_module_available():
            return [sys.executable, "-m", "kaggle.cli"]
        return ["kaggle"]

    def available(self) -> bool:
        return self._cli_path() is not None or self._cli_module_available()

    def _run(
        self, args: Sequence[str], *, check: bool = True
    ) -> subprocess.CompletedProcess:
        argv = [*self._prefix(), *args]
        last: BaseException | None = None
        for attempt in range(1, self._retries + 1):
            try:
                result = subprocess.run(
                    argv, capture_output=True, text=True, timeout=self._timeout
                )
            except subprocess.SubprocessError as exc:
                last = exc
                if attempt == self._retries:
                    raise KaggleError(f"{' '.join(argv)}: {exc}") from exc
                continue
            if check and result.returncode != 0:
                raise KaggleCommandFailed(argv, result.stdout, result.stderr)
            return result
        raise KaggleError(f"{' '.join(argv)}: {last}")

    def _rows(self, args: Sequence[str]) -> list[dict[str, str]]:
        return parse_csv(self._run([*args, "--csv"]).stdout)

    @staticmethod
    def _rejects_page_size(exc: KaggleCommandFailed) -> bool:
        blob = f"{exc.stdout}\n{exc.stderr}".lower()
        return "--page-size" in blob and "unrecognized arguments" in blob

    def _paginated(self, args: Sequence[str]) -> list[dict[str, str]]:
        """Page through a listing, probing ``--page-size`` support once per client.

        Three incompatible policies used to exist: send it and retry without on
        rejection, accept the argument and unconditionally discard it, or send it
        with no fallback. This is the first, generalised.
        """
        rows: list[dict[str, str]] = []
        page = 1
        while True:
            use_page_size = self._page_size_ok is not False
            page_args = list(args)
            if use_page_size:
                page_args += ["--page-size", str(PREFERRED_PAGE_SIZE)]
            page_args += ["--page", str(page)]
            try:
                batch = self._rows(page_args)
            except KaggleCommandFailed as exc:
                if (
                    use_page_size
                    and self._page_size_ok is None
                    and self._rejects_page_size(exc)
                ):
                    self._page_size_ok = False
                    continue
                raise
            if self._page_size_ok is None:
                self._page_size_ok = use_page_size
            expected = PREFERRED_PAGE_SIZE if self._page_size_ok else FALLBACK_PAGE_SIZE
            rows.extend(batch)
            if len(batch) < expected:
                return rows
            page += 1

    def _guard_effects(self, description: str) -> PushOutcome | None:
        if self._effects:
            return None
        self.skipped_effects.append(description)
        return PushOutcome(
            ok=True, detail=f"skipped (effects disabled): {description}", skipped=True
        )

    # -- auth ---------------------------------------------------------------

    @staticmethod
    def _config_path() -> Path:
        config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
        if config_dir:
            return Path(config_dir) / "kaggle.json"
        home_default = Path.home() / ".kaggle" / "kaggle.json"
        if home_default.exists():
            return home_default
        if sys.platform.startswith("linux"):
            xdg = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
            return xdg / "kaggle" / "kaggle.json"
        return home_default

    def credentials(self) -> CredentialState:
        """Resolve credentials once, for everyone.

        Four implementations of this used to disagree: ``KAGGLE_API_TOKEN`` was
        honoured by two of them and ignored by the one that actually reported on
        credential health.
        """
        sources: list[str] = []

        token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
        if token:
            sources.append("environment-token")
            try:
                payload = json.loads(token)
            except json.JSONDecodeError:
                payload = None
            if (
                isinstance(payload, dict)
                and payload.get("username")
                and payload.get("key")
            ):
                return CredentialState(
                    Credentials(
                        str(payload["username"]),
                        str(payload["key"]),
                        "environment-token",
                    ),
                    sources,
                )

        user = os.environ.get("KAGGLE_USERNAME", "").strip()
        key = os.environ.get("KAGGLE_KEY", "").strip()
        if user and key:
            sources.append("environment")
            return CredentialState(Credentials(user, key, "environment"), sources)

        cfg = self._config_path()
        if not cfg.exists():
            if token:
                return CredentialState(
                    Credentials(None, token, "environment-token"), sources
                )
            return CredentialState(
                None, sources, f"Missing Kaggle credentials file: {cfg}"
            )
        sources.append(str(cfg))
        try:
            payload = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return CredentialState(None, sources, f"Invalid kaggle.json: {exc}")
        if not isinstance(payload, dict):
            return CredentialState(None, sources, "kaggle.json must be a JSON object")
        cfg_user = str(payload.get("username", "")).strip()
        cfg_key = str(payload.get("key", "")).strip()
        if not cfg_user or not cfg_key:
            return CredentialState(
                None, sources, "kaggle.json must include non-empty username and key"
            )
        return CredentialState(Credentials(cfg_user, cfg_key, f"file:{cfg}"), sources)

    def probe_upload_auth(self, timeout: int = 30) -> AuthProbe:
        """Check upload authorisation through the Kaggle SDK.

        The import lives inside the method on purpose: kaggle 1.x can call
        ``exit(1)`` at import time when credentials are missing, and as a
        top-level import that was an interpreter-termination hazard for anything
        that merely imported the module holding it.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            from kagglesdk.blobs.types.blob_api_service import (
                ApiBlobType,
                ApiStartBlobUploadRequest,
            )
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - the SDK may exit on import
            return AuthProbe(False, f"Kaggle SDK unavailable: {exc}")
        try:
            api = KaggleApi()
            api.authenticate()
            request = ApiStartBlobUploadRequest()
            request.type = ApiBlobType.DATASET
            request.name = "preflight-probe.csv"
            request.content_length = 1
            request.content_type = "text/csv"
            with api.build_kaggle_client() as client:
                response = client.blobs.blob_api_client.start_blob_upload(request)
            if getattr(response, "create_url", None) and getattr(
                response, "token", None
            ):
                return AuthProbe(True, "upload authorisation granted")
            return AuthProbe(False, "upload probe returned no create_url/token")
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
            return AuthProbe(False, summarize_output(str(exc)))

    # -- kernels ------------------------------------------------------------

    def my_kernels(self, *, kernel_type: str | None = None) -> list[Kernel]:
        args = ["kernels", "list", "--mine"]
        if kernel_type:
            args += ["--kernel-type", kernel_type]
        return [Kernel.from_row(row) for row in self._paginated(args)]

    def push_kernel(self, path: Path) -> PushOutcome:
        skipped = self._guard_effects(f"push kernel {path}")
        if skipped:
            return skipped
        result = self._run(["kernels", "push", "-p", str(path)], check=False)
        ok = result.returncode == 0
        return PushOutcome(
            ok,
            (
                result.stdout if ok else summarize_output(result.stdout, result.stderr)
            ).strip(),
        )

    # -- datasets -----------------------------------------------------------

    def my_datasets(self) -> list[Dataset]:
        return [
            Dataset.from_row(row)
            for row in self._paginated(["datasets", "list", "--mine"])
        ]

    def datasets_by_owner(self, owner: str) -> list[Dataset]:
        """List a user's public datasets.

        Two spellings existed for this — ``-s <owner>`` and ``--user <owner>``.
        ``--user`` is the one that means "owned by", so it is the one we send.
        """
        return [
            Dataset.from_row(row)
            for row in self._rows(["datasets", "list", "--user", owner])
        ]

    def search_datasets(
        self, *, sort_by: str | None = None, pages: int = 1
    ) -> list[Dataset]:
        rows: list[dict[str, str]] = []
        args = ["datasets", "list"]
        if sort_by:
            args += ["--sort-by", sort_by]
        for page in range(1, max(1, pages) + 1):
            rows.extend(self._rows([*args, "--page", str(page)]))
        return [Dataset.from_row(row) for row in rows]

    def dataset_metadata(self, ref: str, dest: Path) -> DatasetMetadata:
        dest.mkdir(parents=True, exist_ok=True)
        self._run(["datasets", "metadata", ref, "-p", str(dest)])
        return read_dataset_metadata(dest / "dataset-metadata.json")

    def dataset_files(self, ref: str) -> list[str]:
        rows = self._rows(["datasets", "files", ref])
        return [
            str(pick(row, ("name",), what="dataset file name")).strip() for row in rows
        ]

    def publish_dataset(self, path: Path, message: str) -> PushOutcome:
        """Version an existing dataset, creating it if it does not exist yet.

        Four call sites hand-rolled this version-then-create fallback.
        """
        skipped = self._guard_effects(f"publish dataset {path}")
        if skipped:
            return skipped
        version = self._run(
            [
                "datasets",
                "version",
                "-p",
                str(path),
                "-m",
                message,
                "--dir-mode",
                "zip",
            ],
            check=False,
        )
        if version.returncode == 0:
            return PushOutcome(True, version.stdout.strip())
        create = self._run(
            ["datasets", "create", "-p", str(path), "--dir-mode", "zip"], check=False
        )
        if create.returncode == 0:
            return PushOutcome(True, create.stdout.strip())
        return PushOutcome(
            False,
            summarize_output(
                version.stdout, version.stderr, create.stdout, create.stderr
            ),
        )

    # -- competitions -------------------------------------------------------

    def entered_competitions(self) -> list[Competition]:
        return [
            Competition.from_row(row)
            for row in self._paginated(["competitions", "list", "--group", "entered"])
        ]

    def search_competitions(
        self, *, category: str | None = None, sort_by: str = "latestDeadline"
    ) -> list[Competition]:
        args = ["competitions", "list", "--sort-by", sort_by]
        if category:
            args += ["--category", category]
        return [Competition.from_row(row) for row in self._rows(args)]

    def leaderboard(self, slug: str) -> list[LeaderboardEntry]:
        rows = self._rows(["competitions", "leaderboard", slug, "--show"])
        return [
            LeaderboardEntry(
                rank=index,
                team_name=str(
                    pick(row, ("teamName", "team"), what="leaderboard team")
                ).strip(),
                score=_as_float(
                    pick(row, ("score",), what="leaderboard score", required=False)
                ),
            )
            for index, row in enumerate(rows, start=1)
        ]

    def submissions(self, slug: str) -> list[Submission]:
        rows = self._rows(["competitions", "submissions", slug])
        return [
            Submission(
                public_score=_as_float(
                    pick(row, ("publicScore",), what="submission score", required=False)
                ),
                raw=row,
            )
            for row in rows
        ]

    def download_competition(self, slug: str, dest: Path) -> Path:
        """Download a competition's data.

        Mutating despite reading: Kaggle may implicitly enter you into the
        competition, so this is gated by ``effects`` like any other write.
        """
        skipped = self._guard_effects(f"download competition {slug}")
        if skipped:
            return dest
        dest.mkdir(parents=True, exist_ok=True)
        self._run(["competitions", "download", "-c", slug, "-p", str(dest), "--force"])
        return dest

    def submit(self, slug: str, file: Path, message: str) -> PushOutcome:
        skipped = self._guard_effects(f"submit {file} to {slug}")
        if skipped:
            return skipped
        result = self._run(
            ["competitions", "submit", "-c", slug, "-f", str(file), "-m", message],
            check=False,
        )
        ok = result.returncode == 0
        return PushOutcome(
            ok,
            (
                result.stdout if ok else summarize_output(result.stdout, result.stderr)
            ).strip(),
        )


def read_dataset_metadata(path: Path) -> DatasetMetadata:
    """Read a ``dataset-metadata.json``, tolerating a double-encoded payload."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise KaggleError(f"{path}: expected a JSON object")
    return DatasetMetadata(
        title=str(payload.get("title", "")),
        subtitle=str(payload.get("subtitle", "")),
        description=str(payload.get("description", "")),
        usability_rating=_as_float(payload.get("usabilityRating")),
        total_votes=_as_int(payload.get("totalVotes")),
        total_downloads=_as_int(payload.get("totalDownloads")),
        is_private=payload.get("isPrivate"),
        keywords=list(payload.get("keywords", []) or []),
        licenses=list(payload.get("licenses", []) or []),
        raw=payload,
    )


# ---------------------------------------------------------------------------
# Test / dry-run adapter
# ---------------------------------------------------------------------------


class FakeKaggleClient:
    """An in-memory Kaggle, seeded with whatever a test needs.

    This ships in the package rather than in ``tests/`` so that
    ``pi-automation``'s suite and any future caller can reach it. Two adapters
    are what make the seam real; one would be indirection.

    Seed with typed objects or with raw CSV rows — rows go through exactly the
    same parsing the production adapter uses, so a fixture captured from real
    Kaggle output (banner and all) exercises the real field resolution.
    """

    def __init__(
        self,
        *,
        kernels: Iterable[Kernel] | None = None,
        datasets: Iterable[Dataset] | None = None,
        competitions: Iterable[Competition] | None = None,
        entered: Iterable[Competition] | None = None,
        leaderboards: dict[str, list[LeaderboardEntry]] | None = None,
        submissions: dict[str, list[Submission]] | None = None,
        metadata: dict[str, DatasetMetadata] | None = None,
        files: dict[str, list[str]] | None = None,
        credentials_state: CredentialState | None = None,
        auth_probe: AuthProbe | None = None,
        available: bool = True,
        effects: bool = True,
        fail_with: KaggleError | None = None,
    ) -> None:
        self._kernels = list(kernels or [])
        self._datasets = list(datasets or [])
        self._competitions = list(competitions or [])
        self._entered = list(entered or [])
        self._leaderboards = dict(leaderboards or {})
        self._submissions = dict(submissions or {})
        self._metadata = dict(metadata or {})
        self._files = dict(files or {})
        # `or` would discard a deliberately-empty state: CredentialState is
        # falsy when it holds no credentials, which is exactly the case a test
        # seeding "no credentials" wants to express.
        self._credentials = (
            credentials_state
            if credentials_state is not None
            else CredentialState(Credentials("tester", "key", "fake"), ["fake"])
        )
        self._auth_probe = (
            auth_probe if auth_probe is not None else AuthProbe(True, "fake")
        )
        self._available = available
        self._effects = effects
        self._fail_with = fail_with
        #: Every mutating call, in order. Assert on this instead of on argv.
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.skipped_effects: list[str] = []

    # -- seeding helpers ----------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        *,
        kernels: str = "",
        datasets: str = "",
        competitions: str = "",
        **kwargs: Any,
    ) -> "FakeKaggleClient":
        """Seed from raw Kaggle CSV text, banner lines included."""
        return cls(
            kernels=[Kernel.from_row(r) for r in parse_csv(kernels)]
            if kernels
            else None,
            datasets=[Dataset.from_row(r) for r in parse_csv(datasets)]
            if datasets
            else None,
            competitions=[Competition.from_row(r) for r in parse_csv(competitions)]
            if competitions
            else None,
            **kwargs,
        )

    def _maybe_fail(self) -> None:
        if self._fail_with is not None:
            raise self._fail_with

    def _record(self, name: str, *args: Any) -> PushOutcome:
        self.calls.append((name, args))
        if not self._effects:
            self.skipped_effects.append(name)
            return PushOutcome(
                True, f"skipped (effects disabled): {name}", skipped=True
            )
        return PushOutcome(True, "ok")

    # -- interface ----------------------------------------------------------

    def available(self) -> bool:
        return self._available

    def credentials(self) -> CredentialState:
        return self._credentials

    def probe_upload_auth(self, timeout: int = 30) -> AuthProbe:
        return self._auth_probe

    def my_kernels(self, *, kernel_type: str | None = None) -> list[Kernel]:
        self._maybe_fail()
        return list(self._kernels)

    def push_kernel(self, path: Path) -> PushOutcome:
        return self._record("push_kernel", path)

    def my_datasets(self) -> list[Dataset]:
        self._maybe_fail()
        return list(self._datasets)

    def datasets_by_owner(self, owner: str) -> list[Dataset]:
        self._maybe_fail()
        return [d for d in self._datasets if d.ref.startswith(f"{owner}/")]

    def search_datasets(
        self, *, sort_by: str | None = None, pages: int = 1
    ) -> list[Dataset]:
        self._maybe_fail()
        return list(self._datasets)

    def dataset_metadata(self, ref: str, dest: Path) -> DatasetMetadata:
        self._maybe_fail()
        if ref not in self._metadata:
            raise KaggleError(f"no seeded metadata for {ref}")
        return self._metadata[ref]

    def dataset_files(self, ref: str) -> list[str]:
        self._maybe_fail()
        return list(self._files.get(ref, []))

    def publish_dataset(self, path: Path, message: str) -> PushOutcome:
        return self._record("publish_dataset", path, message)

    def entered_competitions(self) -> list[Competition]:
        self._maybe_fail()
        return list(self._entered)

    def search_competitions(
        self, *, category: str | None = None, sort_by: str = "latestDeadline"
    ) -> list[Competition]:
        self._maybe_fail()
        if category is None:
            return list(self._competitions)
        # Kaggle matches its category filter case-insensitively: asking for
        # "featured" returns rows whose category reads "Featured".
        wanted = category.strip().lower()
        return [c for c in self._competitions if c.category.strip().lower() == wanted]

    def leaderboard(self, slug: str) -> list[LeaderboardEntry]:
        self._maybe_fail()
        return list(self._leaderboards.get(slug, []))

    def submissions(self, slug: str) -> list[Submission]:
        self._maybe_fail()
        return list(self._submissions.get(slug, []))

    def download_competition(self, slug: str, dest: Path) -> Path:
        self._record("download_competition", slug, dest)
        return dest

    def submit(self, slug: str, file: Path, message: str) -> PushOutcome:
        return self._record("submit", slug, file, message)
