import json
from types import SimpleNamespace

from kaggle_portfolio.ops import leaderboard_tracker as lb

LEADERBOARD_CSV = (
    "Next Page Token = ABC123\n"
    "teamId,teamName,submissionDate,score\n"
    "1,alpha,2026-06-10,0.95\n"
    "2,lorenzoscaturchio,2026-06-11,0.90\n"
    "3,gamma,2026-06-12,0.80\n"
)


class TestParse:
    def test_strips_page_token_and_parses(self):
        rows = lb.parse_leaderboard_csv(LEADERBOARD_CSV)
        assert len(rows) == 3
        assert rows[1]["teamName"] == "lorenzoscaturchio"
        assert rows[0]["score"] == "0.95"

    def test_empty(self):
        assert lb.parse_leaderboard_csv("") == []
        assert lb.parse_leaderboard_csv("Next Page Token = X\n") == []


class TestComputeStanding:
    def setup_method(self):
        self.rows = lb.parse_leaderboard_csv(LEADERBOARD_CSV)

    def test_name_match_rank_and_percentile(self):
        s = lb.compute_standing(self.rows, "lorenzoscaturchio", team_count=3)
        assert s["rank"] == 2
        assert s["team_count"] == 3
        assert s["percentile"] == 66.7
        assert s["score"] == 0.90
        assert s["in_bronze_zone"] is False

    def test_case_insensitive_name(self):
        assert lb.compute_standing(self.rows, "LorenzoScaturchio", team_count=3)["rank"] == 2

    def test_score_fallback_when_name_differs(self):
        s = lb.compute_standing(self.rows, "display-name", owner_scores={0.80}, team_count=3)
        assert s["rank"] == 3
        assert s["score"] == 0.80

    def test_not_found(self):
        s = lb.compute_standing(self.rows, "nobody", team_count=3)
        assert s["rank"] is None
        assert s["percentile"] is None
        assert s["in_bronze_zone"] is False

    def test_bronze_zone_top_fraction(self):
        s = lb.compute_standing(self.rows, "alpha", team_count=100)
        assert s["rank"] == 1
        assert s["in_bronze_zone"] is True

    def test_team_count_defaults_to_row_count(self):
        s = lb.compute_standing(self.rows, "alpha")
        assert s["team_count"] == 3
        assert s["rank"] == 1


class TestFetchers:
    def test_fetch_entered_parses_slug_and_teamcount(self, monkeypatch):
        csv_out = (
            "ref,deadline,category,reward,teamCount,userHasEntered\n"
            "https://www.kaggle.com/competitions/hull-tactical,2026-09-01,Featured,$,1200,True\n"
            "https://www.kaggle.com/competitions/orbit-wars,2026-07-15,Research,Swag,45,True\n"
        )
        monkeypatch.setattr(lb, "_run_csv", lambda args: csv_out)
        assert lb.fetch_entered_competitions() == [
            {"slug": "hull-tactical", "team_count": 1200},
            {"slug": "orbit-wars", "team_count": 45},
        ]

    def test_fetch_entered_handles_cli_failure(self, monkeypatch):
        monkeypatch.setattr(lb, "_run_csv", lambda args: None)
        assert lb.fetch_entered_competitions() == []

    def test_fetch_owner_scores(self, monkeypatch):
        csv_out = (
            "ref,fileName,date,description,status,publicScore,privateScore\n"
            "x,sub1.csv,2026-06-10,first,SubmissionStatus.COMPLETE,0.90,0.91\n"
            "x,sub2.csv,2026-06-11,second,SubmissionStatus.COMPLETE,0.88,\n"
        )
        monkeypatch.setattr(lb, "_run_csv", lambda args: csv_out)
        assert lb.fetch_owner_scores("slug") == {0.90, 0.88}


class TestRecordAndReport:
    def test_cmd_record_dry_run(self, monkeypatch, capsys):
        monkeypatch.setattr(lb, "resolve_credentials",
                            lambda: (SimpleNamespace(username="lorenzoscaturchio"), None))
        monkeypatch.setattr(lb, "fetch_entered_competitions",
                            lambda: [{"slug": "hull-tactical", "team_count": 3}])
        monkeypatch.setattr(lb, "fetch_leaderboard_rows",
                            lambda slug, **k: lb.parse_leaderboard_csv(LEADERBOARD_CSV))
        monkeypatch.setattr(lb, "fetch_owner_scores", lambda slug: set())
        rc = lb.cmd_record(dry_run=True)
        out = capsys.readouterr().out
        assert rc == 0
        assert "DRY RUN" in out
        assert "hull-tactical" in out and "rank=2" in out

    def test_cmd_record_writes_snapshot(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lb, "LEADERBOARD_DIR", tmp_path)
        monkeypatch.setattr(lb, "resolve_credentials",
                            lambda: (SimpleNamespace(username="alpha"), None))
        monkeypatch.setattr(lb, "fetch_entered_competitions",
                            lambda: [{"slug": "c1", "team_count": 3}])
        monkeypatch.setattr(lb, "fetch_leaderboard_rows",
                            lambda slug, **k: lb.parse_leaderboard_csv(LEADERBOARD_CSV))
        monkeypatch.setattr(lb, "fetch_owner_scores", lambda slug: set())
        rc = lb.cmd_record()
        assert rc == 0
        files = list(tmp_path.glob("leaderboard-*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["owner"] == "alpha"
        assert data["standings"][0]["rank"] == 1

    def test_cmd_record_no_credentials(self, monkeypatch):
        monkeypatch.setattr(lb, "resolve_credentials", lambda: (None, "no creds"))
        assert lb.cmd_record() == 1

    def test_build_report_rank_delta(self):
        h = [
            {"generated_on": "2026-06-13",
             "standings": [{"competition": "c1", "rank": 5, "team_count": 100,
                            "percentile": 96.0, "in_bronze_zone": True}]},
            {"generated_on": "2026-06-14",
             "standings": [{"competition": "c1", "rank": 3, "team_count": 100,
                            "percentile": 98.0, "in_bronze_zone": True}]},
        ]
        rep = lb.build_report(h)
        assert rep["generated_on"] == "2026-06-14"
        assert rep["competitions"][0]["rank_delta"] == 2


class TestMain:
    def test_main_record_dispatch(self, monkeypatch):
        called = {}

        def _fake_record(dry_run):
            called["record"] = dry_run
            return 0

        monkeypatch.setattr(lb, "cmd_record", _fake_record)
        assert lb.main(["record", "--dry-run"]) == 0
        assert called["record"] is True

    def test_main_report_dispatch(self, monkeypatch):
        monkeypatch.setattr(lb, "cmd_report", lambda as_json: 0)
        assert lb.main(["report"]) == 0
