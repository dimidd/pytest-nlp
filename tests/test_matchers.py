"""Tests for spaCy matchers module."""

import pytest

from pytest_nlp import (
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

