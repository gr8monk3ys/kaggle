from kaggle_portfolio.datasets import dataset_vote_planner as vp
from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.kaggle_client import (
    Dataset,
    FakeKaggleClient,
    KaggleError,
    parse_csv,
)


class TestNextMedal:
    def test_bronze(self):
        assert vp.next_medal(0) == (5, "bronze", 5)
        assert vp.next_medal(4) == (5, "bronze", 1)

    def test_silver_then_gold(self):
        assert vp.next_medal(5) == (20, "silver", 15)
        assert vp.next_medal(23) == (50, "gold", 27)

    def test_gold_maxed(self):
        threshold, _name, need = vp.next_medal(50)
        assert threshold is None and need == 0
        assert vp.next_medal(80)[2] == 0


class TestScore:
    @staticmethod
    def _meta(slug="spotify-x", keywords=("a", "b")):
        return {"id": f"user/{slug}", "title": "T", "keywords": list(keywords)}

    def test_actions_for_low_votes_and_keywords(self, tmp_path):
        s = vp.score_dataset(
            tmp_path, self._meta(keywords=["a", "b"]), votes=2, downloads=100
        )
        assert s["next_medal"] == "bronze" and s["votes_to_next"] == 3
        assert any("3 more vote" in a for a in s["actions"])
        assert any("keyword" in a for a in s["actions"])  # 2 < 8
        assert any("starter EDA" in a for a in s["actions"])  # no explore.ipynb
        assert any("cover" in a for a in s["actions"])  # no cover.png

    def test_conversion_flag(self, tmp_path):
        s = vp.score_dataset(
            tmp_path, self._meta(keywords=list("abcdefghij")), votes=1, downloads=300
        )
        assert s["conversion_pct"] == round(1 / 300 * 100, 2)
        assert any("low conversion" in a for a in s["actions"])
        assert not any(
            "keyword" in a for a in s["actions"]
        )  # 10 keywords -> no keyword action

    def test_notebook_and_cover_present(self, tmp_path):
        (tmp_path / "explore.ipynb").write_text("{}", encoding="utf-8")
        (tmp_path / "cover.png").write_bytes(b"x")
        s = vp.score_dataset(
            tmp_path, self._meta(keywords=list("abcdefghij")), votes=25, downloads=10
        )
        assert (
            s["has_starter_notebook"] and s["has_cover"] and s["next_medal"] == "gold"
        )
        assert not any("starter" in a or "cover" in a for a in s["actions"])


class TestPlanAndFetch:
    def test_build_plan_orders_by_distance_to_medal(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        ds = [
            (tmp_path / "a", {"id": "u/a", "keywords": []}),
            (tmp_path / "b", {"id": "u/b", "keywords": []}),
        ]
        stats = {"a": {"votes": 4, "downloads": 10}, "b": {"votes": 0, "downloads": 10}}
        plan = vp.build_plan(ds, stats)
        assert [r["slug"] for r in plan] == ["a", "b"]  # a (needs 1) before b (needs 5)

    def test_fetch_live_stats_reads_votes_and_downloads(self):
        client = FakeKaggleClient(
            datasets=[
                Dataset.from_row(row)
                for row in parse_csv(
                    "Warning: outdated API version\n"
                    "ref,title,voteCount,downloadCount\n"
                    "user/spotify-x,Spotify,37,1066\n"
                )
            ]
        )
        assert vp.fetch_live_stats(client, "user") == {
            "spotify-x": {"votes": 37, "downloads": 1066}
        }

    def test_fetch_returns_empty_on_kaggle_failure(self, capsys):
        client = FakeKaggleClient(fail_with=KaggleError("boom"))
        assert vp.fetch_live_stats(client, "user") == {}
        assert "live stats unavailable" in capsys.readouterr().err


def test_main_json_empty(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(vp, "discover_datasets", lambda: [])
    deps = Deps.for_test(tmp_path, client=FakeKaggleClient())
    assert vp.main(["--json"], deps=deps) == 0
    assert capsys.readouterr().out.strip() == "[]"
