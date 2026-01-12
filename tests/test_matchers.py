"""Tests for spaCy matchers module."""

import pytest

from pytest_nlp import (
    ANY,
    assert_matches_dependency,
    assert_matches_phrases,
    assert_matches_tokens,
    match_dependency,
    match_phrases,
    match_tokens,
)


class TestMatchTokens:
    """Tests for token matching."""

    def test_single_token_pattern(self, simple_sentence: str) -> None:
        """Test matching a single token."""
        results = match_tokens(
            doc=simple_sentence,
            patterns=[[{"LOWER": "fox"}]],
        )
        assert len(results) == 1
        assert results[0].text == "fox"

    def test_multi_token_pattern(self, simple_sentence: str) -> None:
        """Test matching multiple tokens."""
        results = match_tokens(
            doc=simple_sentence,
            patterns=[[{"LOWER": "brown"}, {"LOWER": "fox"}]],
        )
        assert len(results) == 1
        assert results[0].text == "brown fox"

    def test_pos_pattern(self, simple_sentence: str) -> None:
        """Test matching by POS tag."""
        results = match_tokens(
            doc=simple_sentence,
            patterns=[[{"POS": "ADJ"}]],
        )
        # Should match "quick", "brown", "lazy"
        assert len(results) == 3

    def test_no_match(self, simple_sentence: str) -> None:
        """Test when pattern doesn't match."""
        results = match_tokens(
            doc=simple_sentence,
            patterns=[[{"LOWER": "elephant"}]],
        )
        assert len(results) == 0

    def test_sentence_filter(self, drug_document: str) -> None:
        """Test filtering by sentence."""
        # "discontinued" only in last sentence
        results = match_tokens(
            doc=drug_document,
            patterns=[[{"LEMMA": "discontinue"}]],
            sentences=[-1],
        )
        assert len(results) == 1


class TestMatchPhrases:
    """Tests for phrase matching."""

    def test_exact_phrase(self, simple_sentence: str) -> None:
        """Test matching exact phrase."""
        results = match_phrases(
            doc=simple_sentence,
            phrases=["quick brown fox"],
        )
        assert len(results) == 1
        assert results[0].text == "quick brown fox"

    def test_multiple_phrases(self, simple_sentence: str) -> None:
        """Test matching multiple phrases."""
        results = match_phrases(
            doc=simple_sentence,
            phrases=["quick brown", "lazy dog"],
        )
        assert len(results) == 2

    def test_case_insensitive(self, simple_sentence: str) -> None:
        """Test case-insensitive matching with LOWER attribute."""
        results = match_phrases(
            doc=simple_sentence,
            phrases=["QUICK BROWN"],
            attr="LOWER",
        )
        assert len(results) == 1

    def test_no_match(self, simple_sentence: str) -> None:
        """Test when phrase doesn't match."""
        results = match_phrases(
            doc=simple_sentence,
            phrases=["pink elephant"],
        )
        assert len(results) == 0


class TestMatchDependency:
    """Tests for dependency matching."""

    def test_subject_verb_pattern(self, simple_sentence: str) -> None:
        """Test matching subject-verb dependency."""
        patterns = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "jump"}},
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "subject",
                "RIGHT_ATTRS": {"DEP": "nsubj"},
            },
        ]
        results = match_dependency(doc=simple_sentence, patterns=patterns)
        assert len(results) >= 1

    def test_nsubjpass_pattern(self, drug_document: str) -> None:
        """Test matching passive subject pattern."""
        patterns = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "discontinue"}},
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "drug",
                "RIGHT_ATTRS": {"DEP": "nsubjpass"},
            },
        ]
        results = match_dependency(doc=drug_document, patterns=patterns)
        assert len(results) >= 1

    def test_sentence_filter(self, drug_document: str) -> None:
        """Test filtering by sentence for dependency patterns."""
        patterns = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "discontinue"}},
        ]
        # Only in last sentence
        results = match_dependency(
            doc=drug_document,
            patterns=patterns,
            sentences=[-1],
        )
        assert len(results) >= 1

        # Not in first sentence
        results_first = match_dependency(
            doc=drug_document,
            patterns=patterns,
            sentences=[0],
        )
        assert len(results_first) == 0


class TestAssertMatches:
    """Tests for assertion functions."""

    def test_assert_tokens_passes(self, simple_sentence: str) -> None:
        """Test that token assertion passes."""
        assert_matches_tokens(
            doc=simple_sentence,
            patterns=[[{"LOWER": "fox"}]],
        )

    def test_assert_tokens_fails(self, simple_sentence: str) -> None:
        """Test that token assertion fails."""
        with pytest.raises(AssertionError) as exc_info:
            assert_matches_tokens(
                doc=simple_sentence,
                patterns=[[{"LOWER": "elephant"}]],
            )
        assert "Token patterns did not match" in str(exc_info.value)

    def test_assert_phrases_passes(self, simple_sentence: str) -> None:
        """Test that phrase assertion passes."""
        assert_matches_phrases(
            doc=simple_sentence,
            phrases=["brown fox"],
        )

    def test_assert_phrases_fails(self, simple_sentence: str) -> None:
        """Test that phrase assertion fails."""
        with pytest.raises(AssertionError) as exc_info:
            assert_matches_phrases(
                doc=simple_sentence,
                phrases=["pink elephant"],
            )
        assert "Phrase patterns did not match" in str(exc_info.value)

    def test_assert_dependency_passes(self, simple_sentence: str) -> None:
        """Test that dependency assertion passes."""
        assert_matches_dependency(
            doc=simple_sentence,
            patterns=[
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "jump"}},
            ],
        )

    def test_assert_dependency_fails(self, simple_sentence: str) -> None:
        """Test that dependency assertion fails."""
        with pytest.raises(AssertionError) as exc_info:
            assert_matches_dependency(
                doc=simple_sentence,
                patterns=[
                    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "fly"}},
                ],
            )
        assert "Dependency patterns did not match" in str(exc_info.value)

    def test_custom_message(self, simple_sentence: str) -> None:
        """Test custom error message."""
        with pytest.raises(AssertionError) as exc_info:
            assert_matches_tokens(
                doc=simple_sentence,
                patterns=[[{"LOWER": "elephant"}]],
                msg="Custom error",
            )
        assert "Custom error" in str(exc_info.value)

    def test_min_matches(self, simple_sentence: str) -> None:
        """Test minimum matches requirement."""
        # Should find 3 adjectives
        assert_matches_tokens(
            doc=simple_sentence,
            patterns=[[{"POS": "ADJ"}]],
            min_matches=3,
        )

        # Should fail if we require more
        with pytest.raises(AssertionError):
            assert_matches_tokens(
                doc=simple_sentence,
                patterns=[[{"POS": "ADJ"}]],
                min_matches=10,
            )


class TestANYSentinelMatchers:
    """Tests for ANY sentinel with matcher functions."""

    def test_match_tokens_with_any(self) -> None:
        """Test token matching with ANY splits on semicolons."""
        text = "First part here; second part contains target word."
        results = match_tokens(
            doc=text,
            patterns=[[{"LOWER": "target"}]],
            sentences=ANY,
        )
        assert len(results) == 1
        assert results[0].text == "target"

    def test_match_phrases_with_any(self) -> None:
        """Test phrase matching with ANY splits on semicolons."""
        text = "Patient stable; medication reduced to 5mg daily."
        results = match_phrases(
            doc=text,
            phrases=["5mg daily"],
            sentences=ANY,
        )
        assert len(results) == 1

    def test_match_phrases_any_case_insensitive(self) -> None:
        """Test case-insensitive phrase matching with ANY."""
        text = "First clause; IMPORTANT PHRASE here."
        results = match_phrases(
            doc=text,
            phrases=["important phrase"],
            sentences=ANY,
            attr="LOWER",
        )
        assert len(results) == 1

    def test_match_dependency_with_any(self) -> None:
        """Test dependency matching with ANY splits on semicolons."""
        text = "Patient rested; doctor prescribed medication."
        patterns = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "prescribe"}},
        ]
        results = match_dependency(
            doc=text,
            patterns=patterns,
            sentences=ANY,
        )
        assert len(results) >= 1

    def test_assert_matches_tokens_with_any(self) -> None:
        """Test token assertion with ANY sentinel."""
        text = "Initial dose; taper initiated later."
        # Should not raise
        assert_matches_tokens(
            doc=text,
            patterns=[[{"LEMMA": "initiate"}]],
            sentences=ANY,
        )

    def test_assert_matches_phrases_with_any(self) -> None:
        """Test phrase assertion with ANY sentinel."""
        text = "Stable vitals; blood pressure normal."
        # Should not raise
        assert_matches_phrases(
            doc=text,
            phrases=["blood pressure"],
            sentences=ANY,
        )

    def test_any_aggregates_results(self) -> None:
        """Test that ANY aggregates results from all sub-sentences."""
        text = "Contains word here; also word there; word again."
        results = match_tokens(
            doc=text,
            patterns=[[{"LOWER": "word"}]],
            sentences=ANY,
        )
        # Should find "word" in all three sub-sentences
        assert len(results) == 3

    def test_complex_medical_text_with_any(self) -> None:
        """Test ANY with complex medical text containing semicolons."""
        text = (
            "Prescribed 20mg daily by Dr. Johnson on 05/01/2024 for an RA flare; "
            "on 05/15/2024, a taper was initiated to 10mg daily for one week, then 5mg daily."
        )
        # Match "taper" in the second sub-sentence
        results = match_tokens(
            doc=text,
            patterns=[[{"LOWER": "taper"}]],
            sentences=ANY,
        )
        assert len(results) == 1
        assert results[0].text == "taper"

