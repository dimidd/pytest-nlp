"""Tests for semantic similarity module."""

import re

import pytest

from pytest_nlp import (
    ANY,
    SentenceMatch,
    assert_semantic_contains,
    assert_semantic_similarity,
    semantic_contains,
    semantic_similarity,
    split_sentences,
)


class TestSemanticSimilarity:
    """Tests for semantic_similarity function."""

    def test_identical_texts(self) -> None:
        """Identical texts should have high similarity."""
        score = semantic_similarity("hello world", "hello world")
        assert score > 0.99

    def test_similar_texts(self) -> None:
        """Semantically similar texts should have reasonable similarity."""
        score = semantic_similarity("The cat sat on the mat", "A cat is sitting on a rug")
        assert score > 0.5

    def test_dissimilar_texts(self) -> None:
        """Dissimilar texts should have low similarity."""
        score = semantic_similarity("The weather is nice today", "Quantum physics is complex")
        assert score < 0.5

    def test_different_metrics(self) -> None:
        """Test different similarity metrics."""
        text1 = "Hello world"
        text2 = "Hi there world"

        cosine = semantic_similarity(text1, text2, metric="cosine")
        dot = semantic_similarity(text1, text2, metric="dot")
        euclidean = semantic_similarity(text1, text2, metric="euclidean")

        # All should return valid scores
        assert 0 <= cosine <= 1
        assert euclidean >= 0  # Euclidean similarity is always positive
        # Dot product can be any value


class TestSemanticContains:
    """Tests for semantic_contains function."""

    def test_contains_high_similarity(self, drug_document: str) -> None:
        """Test finding a phrase with high semantic similarity."""
        is_contained, score = semantic_contains(
            query="Ibuprofen was discontinued",
            document=drug_document,
            threshold=0.7,
        )
        assert is_contained
        assert score > 0.7

    def test_contains_semantic(self, drug_document: str) -> None:
        """Test finding semantically similar content."""
        is_contained, score = semantic_contains(
            query="Ibuprofen was stopped",
            document=drug_document,
            threshold=0.5,
            sentences=slice(-1, None),  # Last sentence only
        )
        assert is_contained, f"Query not semantically contained in document. Score: {score}"
        assert score > 0.5, f"Score is not greater than 0.5. Score: {score}"

    def test_not_contains(self, drug_document: str) -> None:
        """Test content that is not in document."""
        is_contained, score = semantic_contains(
            query="Patient underwent surgery",
            document=drug_document,
            threshold=0.7,
        )
        assert not is_contained
        assert score < 0.7

    def test_sentence_filter_last(self, drug_document: str) -> None:
        """Test filtering to last sentence only."""
        # Discontinuation is in last sentence
        is_contained, score = semantic_contains(
            query="Ibuprofen was discontinued",
            document=drug_document,
            sentences=[-1],
            threshold=0.7,
        )
        assert is_contained

    def test_sentence_filter_first(self, drug_document: str) -> None:
        """Test filtering to first sentence only."""
        # Discontinuation is NOT in first sentence
        is_contained, _ = semantic_contains(
            query="Ibuprofen was discontinued",
            document=drug_document,
            sentences=[0],
            threshold=0.8,
        )
        assert not is_contained

    def test_sentence_filter_slice(self, drug_document: str) -> None:
        """Test filtering with slice."""
        is_contained, _ = semantic_contains(
            query="Ibuprofen was discontinued",
            document=drug_document,
            sentences=slice(-2, None),  # Last 2 sentences
            threshold=0.7,
        )
        assert is_contained


class TestAssertSemanticContains:
    """Tests for assert_semantic_contains function."""

    def test_assertion_passes(self, drug_document: str) -> None:
        """Test that assertion passes for matching content."""
        assert_semantic_contains(
            query="Ibuprofen was discontinued",
            document=drug_document,
            threshold=0.7,
        )

    def test_assertion_fails(self, drug_document: str) -> None:
        """Test that assertion fails for non-matching content."""
        with pytest.raises(AssertionError) as exc_info:
            assert_semantic_contains(
                query="Patient underwent surgery",
                document=drug_document,
                threshold=0.9,
            )
        assert "not semantically contained" in str(exc_info.value)

    def test_custom_message(self, drug_document: str) -> None:
        """Test custom error message."""
        with pytest.raises(AssertionError) as exc_info:
            assert_semantic_contains(
                query="Nonexistent content",
                document=drug_document,
                threshold=0.99,
                msg="Custom error message",
            )
        assert "Custom error message" in str(exc_info.value)


class TestAssertSemanticSimilarity:
    """Tests for assert_semantic_similarity function."""

    def test_assertion_passes(self) -> None:
        """Test that assertion passes for similar texts."""
        assert_semantic_similarity(
            text1="The cat sat on the mat",
            text2="A cat is sitting on a rug",
            threshold=0.5,
        )

    def test_assertion_fails(self) -> None:
        """Test that assertion fails for dissimilar texts."""
        with pytest.raises(AssertionError) as exc_info:
            assert_semantic_similarity(
                text1="Hello world",
                text2="Quantum mechanics",
                threshold=0.9,
            )
        assert "not semantically similar" in str(exc_info.value)


class TestSplitSentences:
    """Tests for split_sentences function."""

    def test_basic_sentence_split(self) -> None:
        """Test basic sentence splitting."""
        text = "Hello world. Goodbye now."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert sentences[0] == "Hello world."
        assert sentences[1] == "Goodbye now."

    def test_semicolon_split(self) -> None:
        """Test splitting on semicolons."""
        text = "Prescribed 20mg daily; later reduced to 10mg."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert sentences[0] == "Prescribed 20mg daily"
        assert sentences[1] == "later reduced to 10mg."

    def test_complex_medical_sentence(self) -> None:
        """Test splitting complex medical text with semicolons."""
        text = (
            "Prescribed 20mg daily by Dr. Johnson on 05/01/2024 for an RA flare; "
            "on 05/15/2024, a taper was initiated to 10mg daily for one week, then 5mg daily."
        )
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert "20mg daily" in sentences[0]
        assert "taper was initiated" in sentences[1]

    def test_no_semicolon_split(self) -> None:
        """Test disabling semicolon splitting."""
        text = "Part one; part two."
        sentences = split_sentences(text, split_on_semicolon=False)
        assert len(sentences) == 1
        assert ";" in sentences[0]

    def test_multiple_semicolons(self) -> None:
        """Test splitting on multiple semicolons."""
        text = "First part; second part; third part."
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_empty_parts_filtered(self) -> None:
        """Test that empty parts are filtered out."""
        text = "Part one;; part two."
        sentences = split_sentences(text)
        assert all(s.strip() for s in sentences)

    def test_combined_periods_and_semicolons(self) -> None:
        """Test text with both periods and semicolons."""
        text = "First sentence. Second part; third part. Fourth sentence."
        sentences = split_sentences(text)
        # Should have: "First sentence.", "Second part", "third part.", "Fourth sentence."
        assert len(sentences) == 4

    def test_comma_and_conjunction_split(self) -> None:
        """Test splitting on comma + 'and'."""
        text = "Prescribed 20mg daily, and later a taper was initiated."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert sentences[0] == "Prescribed 20mg daily"
        assert sentences[1] == "later a taper was initiated."

    def test_comma_but_conjunction_split(self) -> None:
        """Test splitting on comma + 'but'."""
        text = "Patient was stable, but symptoms returned later."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert "stable" in sentences[0]
        assert "symptoms returned" in sentences[1]

    def test_comma_or_conjunction_split(self) -> None:
        """Test splitting on comma + 'or'."""
        text = "Increase the dosage, or consider switching medications."
        sentences = split_sentences(text)
        assert len(sentences) == 2

    def test_complex_medical_with_comma_conjunction(self) -> None:
        """Test complex medical text with comma + conjunction."""
        text = (
            "Prescribed 20mg daily by Dr. Johnson on 05/01/2024 for an RA flare, "
            "and on 05/15/2024, a taper was initiated to 10mg daily"
        )
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert "20mg daily" in sentences[0]
        assert "taper was initiated" in sentences[1]

    def test_no_comma_conjunction_split(self) -> None:
        """Test disabling comma + conjunction splitting."""
        text = "Part one, and part two."
        sentences = split_sentences(text, split_on_comma_conjunction=False)
        assert len(sentences) == 1
        assert ", and" in sentences[0]

    def test_combined_semicolon_and_comma_conjunction(self) -> None:
        """Test text with both semicolons and comma + conjunction."""
        text = "First part; second part, and third part."
        sentences = split_sentences(text)
        # Should have: "First part", "second part", "third part."
        assert len(sentences) == 3

    def test_comma_conjunction_case_insensitive(self) -> None:
        """Test that comma + conjunction splitting is case insensitive."""
        text = "First part, AND second part."
        sentences = split_sentences(text)
        assert len(sentences) == 2

    def test_simple_comma_not_split(self) -> None:
        """Test that simple commas (not followed by conjunction) are not split."""
        text = "The patient, a 45-year-old male, was admitted."
        sentences = split_sentences(text)
        assert len(sentences) == 1
        assert "45-year-old" in sentences[0]


class TestANYSentinel:
    """Tests for ANY sentinel with semantic functions."""

    def test_any_repr(self) -> None:
        """Test ANY sentinel string representation."""
        assert repr(ANY) == "ANY"

    def test_semantic_contains_with_any(self) -> None:
        """Test semantic_contains with ANY splits on semicolons."""
        text = (
            "Prescribed 20mg daily by Dr. Johnson on 05/01/2024 for an RA flare; "
            "on 05/15/2024, a taper was initiated to 10mg daily for one week, then 5mg daily."
        )
        # "taper was initiated" should match the second sub-sentence
        is_contained, score = semantic_contains(
            query="dosage taper initiated",
            document=text,
            sentences=ANY,
            threshold=0.5,
        )
        assert is_contained

    def test_semantic_contains_any_vs_none(self) -> None:
        """Test that ANY can find matches that standard splitting might miss."""
        text = "Patient stable; medication reduced to 5mg."
        
        # With ANY, we get finer granularity
        is_contained_any, score_any = semantic_contains(
            query="medication reduced",
            document=text,
            sentences=ANY,
            threshold=0.5,
        )
        assert is_contained_any

    def test_assert_semantic_contains_with_any(self) -> None:
        """Test assert_semantic_contains with ANY sentinel."""
        text = "Initial dose prescribed; later reduced due to side effects."
        # Should not raise
        assert_semantic_contains(
            query="dose reduced",
            document=text,
            sentences=ANY,
            threshold=0.4,
        )

    def test_assert_semantic_contains_any_fails(self) -> None:
        """Test that assertion fails even with ANY when content not present."""
        text = "Patient was admitted; blood work ordered."
        with pytest.raises(AssertionError) as exc_info:
            assert_semantic_contains(
                query="surgery performed",
                document=text,
                sentences=ANY,
                threshold=0.9,
            )
        assert "not semantically contained" in str(exc_info.value)


class TestSentenceMatch:
    """Tests for SentenceMatch pattern-based sentence selection."""

    def test_matches_string_case_insensitive(self) -> None:
        """Test that string matching is case insensitive."""
        matcher = SentenceMatch("medication")
        assert matcher.matches("Patient received MEDICATION today.")
        assert matcher.matches("medication was prescribed")
        assert not matcher.matches("Patient was admitted.")

    def test_matches_regex(self) -> None:
        """Test matching with compiled regex pattern."""
        matcher = SentenceMatch(re.compile(r"\d+mg"))
        assert matcher.matches("Prescribed 20mg daily.")
        assert matcher.matches("Later reduced to 5mg.")
        assert not matcher.matches("No dosage specified.")

    def test_select_first(self) -> None:
        """Test selecting first matching sentence."""
        sentences = [
            "Patient was admitted.",
            "Prescribed ibuprofen 200mg.",
            "Later increased to 400mg.",
            "Patient discharged.",
        ]
        matcher = SentenceMatch(re.compile(r"\d+mg"), mode="first")
        result = matcher.select(sentences)
        assert result == ["Prescribed ibuprofen 200mg."]

    def test_select_last(self) -> None:
        """Test selecting last matching sentence."""
        sentences = [
            "Patient was admitted.",
            "Prescribed ibuprofen 200mg.",
            "Later increased to 400mg.",
            "Patient discharged.",
        ]
        matcher = SentenceMatch(re.compile(r"\d+mg"), mode="last")
        result = matcher.select(sentences)
        assert result == ["Later increased to 400mg."]

    def test_select_all(self) -> None:
        """Test selecting all matching sentences."""
        sentences = [
            "Patient was admitted.",
            "Prescribed ibuprofen 200mg.",
            "Later increased to 400mg.",
            "Patient discharged.",
        ]
        matcher = SentenceMatch(re.compile(r"\d+mg"), mode="all")
        result = matcher.select(sentences)
        assert result == ["Prescribed ibuprofen 200mg.", "Later increased to 400mg."]

    def test_select_no_match(self) -> None:
        """Test that empty list is returned when nothing matches."""
        sentences = ["Patient was admitted.", "Patient discharged."]
        matcher = SentenceMatch("medication", mode="all")
        result = matcher.select(sentences)
        assert result == []

    def test_semantic_contains_with_sentence_match_first(
        self, drug_document: str
    ) -> None:
        """Test semantic_contains with SentenceMatch selecting first match."""
        # Find first sentence containing "Ibuprofen"
        is_contained, score = semantic_contains(
            query="drug prescription",
            document=drug_document,
            sentences=SentenceMatch("ibuprofen", mode="first"),
            threshold=0.3,
        )
        assert is_contained

    def test_semantic_contains_with_sentence_match_last(
        self, drug_document: str
    ) -> None:
        """Test semantic_contains with SentenceMatch selecting last match."""
        # Find last sentence mentioning Ibuprofen (the discontinuation sentence)
        is_contained, score = semantic_contains(
            query="Ibuprofen discontinued",
            document=drug_document,
            sentences=SentenceMatch("ibuprofen", mode="last"),
            threshold=0.6,
        )
        assert is_contained

    def test_semantic_contains_with_sentence_match_all(
        self, drug_document: str
    ) -> None:
        """Test semantic_contains with SentenceMatch selecting all matches."""
        is_contained, score = semantic_contains(
            query="Ibuprofen treatment",
            document=drug_document,
            sentences=SentenceMatch("ibuprofen", mode="all"),
            threshold=0.4,
        )
        assert is_contained

    def test_assert_semantic_contains_with_sentence_match(
        self, drug_document: str
    ) -> None:
        """Test assert_semantic_contains with SentenceMatch."""
        # Should not raise - matching "Ibuprofen was discontinued" sentence
        assert_semantic_contains(
            query="Ibuprofen was discontinued",
            document=drug_document,
            sentences=SentenceMatch("discontinue", mode="first"),
            threshold=0.7,
        )

    def test_sentence_match_no_match_returns_false(self, drug_document: str) -> None:
        """Test that no matching sentences returns (False, 0.0)."""
        is_contained, score = semantic_contains(
            query="surgery performed",
            document=drug_document,
            sentences=SentenceMatch("nonexistent_word_xyz"),
            threshold=0.5,
        )
        assert not is_contained
        assert score == 0.0

    def test_sentence_match_regex_dosage_first(self, dosage_document: str) -> None:
        """Test SentenceMatch with regex pattern for first dosage sentence."""
        # Find first sentence containing a dosage (e.g., "200mg", "400mg")
        is_contained, score = semantic_contains(
            query="medication prescribed",
            document=dosage_document,
            sentences=SentenceMatch(re.compile(r"\d+mg"), mode="first"),
            threshold=0.3,
        )
        assert is_contained

    def test_sentence_match_regex_dosage_last(self, dosage_document: str) -> None:
        """Test SentenceMatch with regex pattern for last dosage sentence."""
        # Find last sentence containing a dosage
        is_contained, score = semantic_contains(
            query="dosage increased",
            document=dosage_document,
            sentences=SentenceMatch(re.compile(r"\d+mg"), mode="last"),
            threshold=0.5,
        )
        assert is_contained

    def test_sentence_match_regex_dosage_all(self, dosage_document: str) -> None:
        """Test SentenceMatch with regex pattern for all dosage sentences."""
        # Find all sentences containing dosages
        is_contained, score = semantic_contains(
            query="medication dosage",
            document=dosage_document,
            sentences=SentenceMatch(re.compile(r"\d+mg"), mode="all"),
            threshold=0.3,
        )
        assert is_contained

    def test_sentence_match_regex_select_returns_correct_sentences(
        self, dosage_document: str
    ) -> None:
        """Test that regex SentenceMatch selects the correct sentences."""
        from pytest_nlp import split_sentences

        sentences = split_sentences(dosage_document, split_on_semicolon=False)
        matcher = SentenceMatch(re.compile(r"\d+mg"), mode="all")
        selected = matcher.select(sentences)

        # Should select the two sentences with dosages
        assert len(selected) == 2
        assert "200mg" in selected[0]
        assert "400mg" in selected[1]

