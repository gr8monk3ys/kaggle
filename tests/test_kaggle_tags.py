"""Kaggle applies only keywords from its own 833-tag vocabulary."""

from __future__ import annotations

from kaggle_portfolio.shared import kaggle_tags


class TestVocabulary:
    def test_the_akkadian_rejection_is_reproduced_exactly(self):
        """The one push whose rejection we actually observed, replayed offline.

        Pushing akkadian-translation-sentence-match-baseline printed:
          "not valid tags and could not be added: ['machine translation',
           'akkadian', 'ancient languages', 'retrieval', 'sentence alignment']"
        accepting only `nlp`. If this oracle disagrees with that, it is wrong.
        """
        declared = [
            "nlp",
            "machine translation",
            "akkadian",
            "ancient languages",
            "retrieval",
            "sentence alignment",
        ]
        assert kaggle_tags.invalid_tags(declared) == declared[1:]
        assert kaggle_tags.is_valid_tag("nlp")

    def test_the_terms_a_previous_pass_deleted_are_the_real_ones(self):
        """PR #78 dropped these as 'generic filler'. They are the actual vocabulary."""
        for keyword in ("education", "eda", "deep learning", "beginner", "kaggle"):
            assert kaggle_tags.is_valid_tag(keyword), keyword

    def test_the_terms_it_kept_as_searchable_do_not_exist(self):
        """...while the 'specific, searchable' ones it preserved are not tags at all."""
        for keyword in ("lora", "qlora", "shap", "polars", "duckdb"):
            assert not kaggle_tags.is_valid_tag(keyword), keyword

    def test_matching_ignores_case_and_surrounding_space(self):
        assert kaggle_tags.is_valid_tag("  Deep Learning  ")

    def test_slugs_resolve_as_well_as_names(self):
        assert kaggle_tags.is_valid_tag("model-explainability")
        assert kaggle_tags.is_valid_tag("model explainability")

    def test_suggest_offers_a_real_tag_for_a_rejected_one(self):
        assert "model explainability" in kaggle_tags.suggest("explain")
        assert kaggle_tags.suggest("time series") == ["time series analysis"]

    def test_suggest_is_empty_when_the_concept_has_no_tag(self):
        assert kaggle_tags.suggest("qlora") == []

    def test_the_applied_cap_is_five(self):
        """Observed, not documented: no live notebook or dataset carries more than 5."""
        assert kaggle_tags.MAX_APPLIED_KEYWORDS == 5

    def test_the_vocabulary_actually_loaded(self):
        """A missing or truncated TSV must fail loudly, not pass everything."""
        assert len(kaggle_tags._vocabulary()) > 1000  # 833 tags, name + slug each
