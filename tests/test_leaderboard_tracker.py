import json

import pytest

from kaggle_portfolio.ops import leaderboard_tracker as lb
from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import (
    Competition,
    CredentialState,
    Credentials,
    FakeKaggleClient,
    KaggleError,
    LeaderboardEntry,
    Submission,
    parse_csv,
)

# Real `competitions leaderboard --show --csv` output, page-token line included.
LEADERBOARD_CSV = (
    "Next Page Token = ABC123\n"
    "teamId,teamName,submissionDate,score\n"
    "1,alpha,2026-06-10,0.95\n"
    "2,lorenzoscaturchio,2026-06-11,0.90\n"
    "3,gamma,2026-06-12,0.80\n"
)

BOARD = [
    LeaderboardEntry(1, "alpha", 0.95),
    LeaderboardEntry(2, "lorenzoscaturchio", 0.90),
    LeaderboardEntry(3, "gamma", 0.80),
]

OWNER = Credentials("lorenzoscaturchio", "key", "fake")


def _client(**kwargs) -> FakeKaggleClient:
    kwargs.setdefault("credentials_state", CredentialState(OWNER, ["fake"]))
    return FakeKaggleClient(**kwargs)


def _deps(tmp_path, client, *, effects=True, today="2026-06-11") -> Deps:
    return Deps.for_test(tmp_path, today=today, client=client, effects=effects)


class TestLeaderboardParsing:
    """The client owns parsing; these pin the shape the tracker relies on."""

    def test_page_token_line_is_stripped(self):
        rows = parse_csv(LEADERBOARD_CSV)
        assert [r["teamName"] for r in rows] == ["alpha", "lorenzoscaturchio", "gamma"]

    def test_empty_and_token_only_output(self):
        assert parse_csv("") == []
        assert parse_csv("Next Page Token = X\n") == []


class TestComputeStanding:
    def test_name_match_rank_and_percentile(self):
        s = lb.compute_standing(BOARD, "lorenzoscaturchio", team_count=3)
        assert s["rank"] == 2
        assert s["team_count"] == 3
        assert s["percentile"] == 66.7
        assert s["score"] == 0.90
        assert s["in_bronze_zone"] is False

    def test_case_insensitive_name(self):
        assert (
            lb.compute_standing(BOARD, "LorenzoScaturchio", team_count=3)["rank"] == 2
        )

    def test_score_fallback_when_name_differs(self):
        s = lb.compute_standing(
            BOARD, "display-name", owner_scores={0.80}, team_count=3
        )
        assert s["rank"] == 3
        assert s["score"] == 0.80

    def test_not_found(self):
        s = lb.compute_standing(BOARD, "nobody", team_count=3)
        assert s["rank"] is None
        assert s["percentile"] is None
        assert s["in_bronze_zone"] is False

    def test_bronze_zone_top_fraction(self):
        s = lb.compute_standing(BOARD, "alpha", team_count=100)
        assert s["rank"] == 1
        assert s["in_bronze_zone"] is True

    def test_team_count_defaults_to_row_count(self):
        assert lb.compute_standing(BOARD, "alpha")["team_count"] == 3


class TestFetches:
    def test_fetch_entered_parses_slug_and_teamcount(self):
        client = _client(
            entered=[
                Competition.from_row(r)
                for r in parse_csv(
                    "ref,title,teamCount\nkaggle/titanic,Titanic,14000\n"
                )
            ]
        )
        assert lb.fetch_entered_competitions(client) == [
            {"slug": "titanic", "team_count": 14000}
        ]

    def test_fetch_entered_tolerates_kaggle_failure(self, capsys):
        client = _client(fail_with=KaggleError("boom"))
        assert lb.fetch_entered_competitions(client) == []
        assert "kaggle call failed" in capsys.readouterr().err

    def test_fetch_owner_scores(self):
        client = _client(
            submissions={
                "titanic": [Submission(0.8), Submission(None), Submission(0.9)]
            }
        )
        assert lb.fetch_owner_scores(client, "titanic") == {0.8, 0.9}


class TestRecord:
    def _entered(self):
        return [
            Competition.from_row(r)
            for r in parse_csv("ref,title,teamCount\nkaggle/titanic,T,3\n")
        ]

    def test_record_writes_a_snapshot_named_from_the_clock(self, tmp_path):
        client = _client(
            entered=self._entered(),
            leaderboards={"titanic": BOARD},
            submissions={"titanic": [Submission(0.90)]},
        )
        deps = _deps(tmp_path, client)
        assert lb.cmd_record(deps) == 0
        written = sorted(deps.layout.leaderboard_dir.glob("leaderboard-*.json"))
        assert [p.name for p in written] == ["leaderboard-2026-06-11T000000Z.json"]
        snapshot = json.loads(written[0].read_text())
        assert snapshot["generated_on"] == "2026-06-11"
        assert snapshot["standings"][0]["rank"] == 2

    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        client = _client(
            entered=self._entered(),
            leaderboards={"titanic": BOARD},
            submissions={"titanic": []},
        )
        deps = _deps(tmp_path, client, effects=False)
        assert lb.cmd_record(deps) == 0
        assert "DRY RUN" in capsys.readouterr().out
        assert not deps.layout.leaderboard_dir.exists()

    def test_record_without_credentials(self, tmp_path, capsys):
        client = _client(credentials_state=CredentialState(None, [], "no credentials"))
        assert lb.cmd_record(_deps(tmp_path, client)) == 1
        assert "Cannot resolve Kaggle username" in capsys.readouterr().err

    def test_record_with_no_entered_competitions(self, tmp_path):
        assert lb.cmd_record(_deps(tmp_path, _client(entered=[]))) == 0


class TestReport:
    def test_build_report_rank_delta(self):
        history = [
            {
                "generated_on": "2026-06-01",
                "standings": [{"competition": "t", "rank": 5}],
            },
            {
                "generated_on": "2026-06-02",
                "standings": [{"competition": "t", "rank": 3}],
            },
        ]
        report = lb.build_report(history)
        assert report["competitions"][0]["rank_delta"] == 2

    def test_report_with_no_history(self, tmp_path, capsys):
        assert lb.cmd_report(_deps(tmp_path, _client())) == 0
        assert "No leaderboard history yet" in capsys.readouterr().out


class TestMain:
    def test_main_record_dispatch(self, tmp_path):
        deps = _deps(tmp_path, _client(entered=[]))
        assert lb.main(["record"], deps=deps) == 0

    def test_main_report_dispatch(self, tmp_path, capsys):
        deps = _deps(tmp_path, _client())
        assert lb.main(["report", "--json"], deps=deps) == 0

    def test_main_requires_a_subcommand(self, tmp_path):
        # argparse exits by design when a required subcommand is missing; that is
        # not a CommandError, and the dispatcher deliberately does not catch it.
        with pytest.raises(SystemExit):
            lb.main([], deps=_deps(tmp_path, _client()))
